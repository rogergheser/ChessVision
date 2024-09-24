import chess
import os
import sys
import math
import logging
import numpy as np
import threading
import queue
import time
import rembg
sys.path.append('./chesscog')
from data_processing.fens_from_png import fens_from_pgn
from data_processing.rename import parse_for_sorting
from chesscog.core.exceptions import ChessboardNotLocatedException
from chesscog.recognition.recognition import ChessRecognizer
from chesscog.occupancy_classifier.download_model import ensure_model as ensure_occupancy_classifier
from chesscog.piece_classifier.download_model import ensure_model as ensure_piece_classifier
from recap import URI
from pathlib import Path
from chess import Status
from PIL import Image
from highlight_moves import *

DATADIR = 'data/chess_games'
PGN_DIR = 'data/PGNS/chess_games'
MAX_RETRIES = 6
BOARD_SIZE = 400
MAX_QUEUE_LEN = 40
MAX_ERRORS = 6
failed_execution = False
image_queue = queue.Queue()

def display_images():
    while not failed_execution:
        if not image_queue.empty():
            image = image_queue.get()
            cv2.imshow('Image', image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            time.sleep(0.3)

    cv2.destroyAllWindows()

def get_dataloader(datadir, pgn_dir):
    """
    :param: datadir: str - path to directory containing images
    :param: pgn_dir: str - path to dir containing pgn files containing the chess games. These act as the ground truths
    for the images
    This function scans the data directory for images and optionally pgn files that act as ground truth.
    It then returns a dictionary containing Game Name, PGN File, Image File paths for each game with a corresponding FEN.
    """
    res = {}

    for dir in os.listdir(datadir):
        if dir == '.DS_Store':
            continue
        res[dir] = {'name': dir, 'images': []}
        fens = None
        if pgn_dir is not None:
            pgn_file = '{}.pgn'.format(os.path.join(pgn_dir, dir))
            if os.path.exists(pgn_file):
                res[dir]['pgn'] = pgn_file
                fens = fens_from_pgn(pgn_file)
            else:
                raise FileNotFoundError('No PGN file found for game {}\n'.format(pgn_file))
        if pgn_dir is not None and len(fens) != len(os.listdir(os.path.join(datadir, dir))):
            raise ValueError('Number of FENs and images do not match for game {}\n FENS:{}-Images:{}'
                             .format(dir, len(fens), len(os.listdir(os.path.join(datadir, dir)))))
        files = sorted(os.listdir(os.path.join(datadir, dir)), key=lambda x : parse_for_sorting(x))
        for idx, image in enumerate(files):
            image_path = os.path.join(datadir, dir, image)
            if pgn_dir:
                res[dir]['images'].append((image_path, fens[idx]))
            else:
                res[dir]['images'].append(image_path)

    return res

def valid_move(diff_map: list, board1: chess.Board, board2: chess.Board):
    if len(diff_map) != 2:
        return False
    
    
    if diff_map[0][2] == 'empty':
        start_square = diff_map[0][0]
        end_square = diff_map[1][0]
        moving_piece = diff_map[0][1]
        landed_piece = diff_map[1][2]
    else:
        start_square = diff_map[1][0]
        end_square = diff_map[0][0]
        moving_piece = diff_map[1][1]
        landed_piece = diff_map[0][2]
    
    # TODO CHECK THIS
    return board1.is_legal(chess.Move(start_square, end_square)) and moving_piece == landed_piece

def valid_move_diff_piece(diff_map: list, board1: chess.Board, board2: chess.Board):
    """
    Checks if the move is valid if the landing piece or starting piece is different
    :param: diff_map: dict - dictionary containing the differences between the two board states
    :param: board1: chess.Board - board state at time t
    :param: board2: chess.Board - board state at time t+1
    """

    if len(diff_map) != 2:
        return False
    
    if diff_map[0][2] == 'empty':
        start_square = diff_map[0][0]
        end_square = diff_map[1][0]
        moving_piece = diff_map[0][1]
        landing_square_piece = diff_map[1][2]
    else:
        start_square = diff_map[1][0]
        end_square = diff_map[0][0]
        moving_piece = diff_map[1][1]
        landing_square_piece = diff_map[0][2]

        diff_map[0], diff_map[1] = diff_map[1], diff_map[0]

    if board1.is_legal(chess.Move(start_square, end_square)):
        if moving_piece != board1.piece_at(start_square).symbol():
            raise ValueError('Piece leaving from square {} mismatch. Detected "{}" expected "{}".\nExiting...'
                                .format(start_square, moving_piece, board1.piece_at(start_square).symbol()))

        board2 = board1.copy()
        board2.push(chess.Move(start_square, end_square))
        diff_map[1] = (diff_map[1][0], moving_piece, diff_map[1][2])
        return True, diff_map, board1, board2
    else:
        return False, diff_map, board1, board2

def appearing_piece(diff_map: list, board1: chess.Board, board2: chess.Board):
    """
    :param: diff_map: list - list of tuples (square, piece1, piece2)
    :param: board1: chess.Board - board state at time t
    :param: board2: chess.Board - board state at time t+1
    This function checks if a piece has appeared on the board and if so, checks if the move is valid/legal.
    If there is no way for a piece to appear in such square the piece is removed.
    """
    changes = []
    appearances = []
    vacating = []
    for square, piece1, piece2 in diff_map:
        if piece1 == 'empty' and piece2 != 'empty':
            # TODO evaluate whether to extend this even for illegal captures
            appearances.append((square, piece2))
        if piece1 != 'empty' and piece2 == 'empty':
            vacating.append((square, piece1))
    
    if len(vacating) == 1: 
        for square, piece in appearances:
            if not board1.is_legal(chess.Move(vacating[0][0], square)):
                changes.append((square, piece, 'empty'))
                logger.debug('Piece {} appeared on square {} and no legal move was found to place it there. Removing piece.\n'
                             .format(piece, square))

    for appearance in appearances:
        algebraic_move = chess.Move

    return changes, board1, board2

def check_castling(diff_map: list, board1: chess.Board, board2: chess.Board):
    """
    :param: diff_map: list - list of tuples (square, piece1, piece2)
    :param: board1: chess.Board - board state at time t
    :param: board2: chess.Board - board state at time t+1
    This function checks if the diff_map is a valid castling move
    """
    white_turn = board1.turn
    king = chess.Piece.from_symbol('K' if white_turn else 'k')
    if white_turn:
        if board1.piece_at(chess.E1) == king and board2.piece_at(chess.G1) == king and \
            chess.Move(chess.E1, chess.G1) in board1.legal_moves:
            squares = [chess.E1, chess.F1, chess.G1, chess.H1]
        elif board1.piece_at(chess.E1) == king and board2.piece_at(chess.C1) == king and \
            chess.Move(chess.E1, chess.C1) in board1.legal_moves:
            squares = [chess.E1, chess.D1, chess.C1, chess.A1]  
        else:
            return diff_map, board1, board2
    else:
        if board1.piece_at(chess.E8) == king and board2.piece_at(chess.G8) == king and \
            chess.Move(chess.E8, chess.G8) in board1.legal_moves:
            squares = [chess.E8, chess.F8, chess.G8, chess.H8]
        elif board1.piece_at(chess.E8) == king and board2.piece_at(chess.C8) == king and \
            chess.Move(chess.E8, chess.C8) in board1.legal_moves:
            squares = [chess.E8, chess.D8, chess.C8, chess.A8]
        else:
            return diff_map, board1, board2

    for castle_square in squares:
        if castle_square not in [square for square, _, _ in diff_map]:
            return diff_map, board1, board2

    diff_map = [diff for diff in diff_map if diff[0] not in squares]
    diff_map.append((squares[0], 'K' if white_turn else 'k', 'empty'))
    diff_map.append((squares[2], 'R' if white_turn else 'r', 'empty'))
    board2 = board1.copy().push(chess.Move(squares[0], squares[2]))
    
    return diff_map, board1, board2

def check_en_passant(diff_map: list, board1: chess.Board, board2: chess.Board):
    """
    :param: diff_map: list - list of tuples (square, piece1, piece2)
    :param: board1: chess.Board - board state at time t
    :param: board2: chess.Board - board state at time t+1
    This function checks if the diff_map is a valid en passant move
    """
    if len(diff_map) != 3:
        return diff_map, board1, board2

    white_turn = board1.turn
    
    en_passant_move = board1.fen().split(' ')[3]
    if en_passant_move == '-':
        return diff_map, board1, board2
    else:
        square_idx = chess.parse_square(en_passant_move)
        for square, piece1, piece2 in diff_map:
            if (piece1 == 'P' or piece1 == 'p') and piece2 == 'empty':
                starting_square = square
            if piece1 != 'empty' and piece2 == 'empty':
                removed_piece_square = square
            if piece1 == 'empty' and (piece2 == 'p' or piece2 == 'P'):
                landing_square = square

        diff_map = [diff for diff in diff_map if diff[0] != removed_piece_square]

    return diff_map, board1, board2

def pawn_constraints(diff_map: list, board1: chess.Board, board2: chess.Board):
    """
    Pawns are not allowed to be on the first or last rank. If they are, they are removed.
    """

    logger.debug("[WARNING]Bugged function consider removing before running")

    changes = []
    for idx, (square, piece1, piece2) in enumerate(diff_map):
        if (piece2 == 'p' or piece2 == 'P') and chess.square_rank(square) in [0, 7]:
            changes.append((square, piece1, piece2))
            
            if piece1 != 'empty':
                diff_map[idx] = (square, piece1, 'empty')
            else:
                diff_map.pop(idx)

            logger.debug('Pawn {} is not allowed to be on the first or last rank. Removing piece.\n'.format(piece2))    
            # remove the pawns from board2
            if board2.remove_piece_at(square) is None:
                raise ValueError('Failed to remove piece at square {} from board2\nExiting...'.format(square))

    return changes, board1, board2


def polish_results(board1: chess.Board, board2: chess.Board, path1, path2, white_turn):
    """
    :param: board1: chess.Board - board state at time t
    :param: board2: chess.Board - board state at time t+1
    This function polishes the results by comparing the two board states and returning the diff_map, and the two board states
    """
    diff_map = get_diff_map(board1, board2)
    
    if len(diff_map) == 0:
        print("debug")

    filters = [same_color_diff_piece, appearing_piece, pawn_constraints]

    # Apply constraints to clean up the results
    for filter in filters:
        changes, board1, board2 = filter(diff_map, board1, board2)
        
        if len(changes) > 0:
            for change in changes:
                try:
                    diff_map.remove(change)
                except:
                    print('Failed to remove change from diff_map')
                    print(diff_map)
                    print(change)

    if len(diff_map) > 3:
    # e.g. a piece appeared randomly on a square
    # TODO - Implement a more sophisticated algorithm to handle this case
    # If not check if the diff_map is a valid promotion move
        if board1.has_castling_rights(board1.turn):
            diff_map, board1, board2 = check_castling(diff_map, board1, board2)
    elif len(diff_map) == 3:
        diff_map, board1, board2 = check_en_passant(diff_map, board1, board2)
    elif len(diff_map) == 2:
        if valid_move(diff_map, board1, board2):
            return diff_map, board1, board2
        else:
            res, new_diff_map, new_board1, new_board2 = valid_move_diff_piece(diff_map, board1, board2)
            if res:
                diff_map = new_diff_map
                board1 = new_board1
                board2 = new_board2
    elif len(diff_map) == 0:
        if board1().fen() == board2.fen():
            raise ValueError('No differences detected between the two board states.\nExiting...')
        # todo handle this case
        pass
    else:
        raise ValueError('Invalid board states detected. Only {} squares differ between the two board states.\n{}\n{}Exiting...'
                         .format(len(diff_map), path1, path2))
    
    return diff_map, board1, board2

def add_noise(image, patience):
    """
    :param: image: PIL.Image - image to add noise to
    """
    image_array = np.array(image)

    mean = 0
    std = 1 * math.sqrt(MAX_RETRIES-patience)
    
    logger.debug("Introducing gaussian noise with std:{} to image\n".format(std))

    gauss_noise = np.random.normal(mean, std, image_array.shape)

    noisy_image = np.clip(image_array + gauss_noise, 0, 255).astype(np.uint8)

    return noisy_image

def parse_turn(path):
    char = path.split('.')[0][-1]
    return True if char == 'W' else False

def test_rejection(recognizer, board1, board2, image_path):
    image2 = Image.open(image_path)
    patience = MAX_RETRIES

    while patience > 0 and not acceptable_diff(board1, board2):
        logger.debug("Retrying inference for {} with extra noise and patience:{}\n".format(image_path, patience))
        
        image2 = cv2.imread(image_path)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        # add some random noise to the image
        image2 = add_noise(image2, patience)
        board2, *_ = recognizer.predict(image2, parse_turn(image_path))
        patience -= 1

    if patience == 0:
        logger.debug('Failed to converge to a solution for {}.\nPlease change the image or fine tune the model better.\nExiting...')
        raise ValueError('Failed to converge to a solution for {}.\nPlease change the image or fine tune the model better.\nExiting...'
                         .format(image_path))
    logger.debug('Successfully converged to a solution for {}.\n'.format(image_path))

    return board2

def acceptable_diff(board1, board2):    
    return len(get_diff_map(board1, board2)) < MAX_ERRORS

def get_diff_map(board1, board2):
    """
    :param: board1: chess.Board - board state at time t
    :param: board2: chess.Board - board state at time t+1
    This function returns the differences between the two board states
    """
    diff_map = []
    board_map1, board_map2 = board1.piece_map(), board2.piece_map()
    for i in range(64):
        if i not in board_map1.keys():
            board_map1[i] = EmptySquare()
        if i not in board_map2.keys():
            board_map2[i] =  EmptySquare()
        
        if board_map1[i] != board_map2[i]:
            diff_map.append((i, board_map1[i].symbol(), board_map2[i].symbol()))

    return diff_map

def create_final_frame(b1_frame, old_b2_frame, b2_frame, move_frame, image1, image2):
    """
    Takes the four frames and joins them into a single image with labels for each frame
    """
    from PIL import Image, ImageFont, ImageDraw
    size = BOARD_SIZE
    final_frame = Image.new('RGB', (3*size, 2*size))
    final_frame.paste(svg_to_image((b1_frame)), (0, 0))
    final_frame.paste(svg_to_image((old_b2_frame)), (size, 0))
    final_frame.paste(svg_to_image((b2_frame)), (0, size))
    final_frame.paste(svg_to_image((move_frame)), (size, size))
    final_frame.paste(Image.fromarray(image1).resize((400, 400)), (2*size, 0))
    final_frame.paste(Image.fromarray(image2).resize((400, 400)), (2*size, size))
    # TODO ADD LABELS

    final_frame = PIL2cv2(final_frame)

    return final_frame

def visualise_results(board1, old_board2, board2, image1, image2, diff_map):
    """
    :param: board1: chess.Board - image of the board state at time t
    :param: old_board2: chess.Board - image of the board state at time t+1 before being polished
    :param: board2: chess.Board - image of the board state at time t+1
    This function visualises the results by highlighting the differences between the two board states.
    """

    
    polishing_diff = get_diff_map(old_board2, board2)
    fill = {square: 'red' for square, _, _ in polishing_diff}

    b1_frame = chess.svg.board(board1)
    old_b2_frame = chess.svg.board(old_board2, fill=fill)
    b2_frame = chess.svg.board(board2)

    if len(diff_map) == 2:
        if diff_map[0][2] == 'empty':
            start_square, end_square = diff_map[0][0], diff_map[1][0]
        else:
            start_square, end_square = diff_map[1][0], diff_map[0][0]
        move = chess.Move(start_square, end_square)
        move_frame = chess.svg.board(board2, fill={}, arrows=[chess.svg.Arrow(start_square, end_square)], size=BOARD_SIZE)
    else:
        fill = get_square_fill(diff_map)
        move_frame = chess.svg.board(board1, fill=fill, size=BOARD_SIZE)

    final_frame = create_final_frame(b1_frame, old_b2_frame, b2_frame, move_frame, image1, image2)

    return final_frame


def process_game(game, classifier_folder):
    """
    :param: game: dict - contains the game name, pgn file, and image file paths for the game
    This function processes a game by running inference on the images and saving the results.
    """
    recognizer = ChessRecognizer(classifier_folder)
    image_path1 = game['images'][0]
    if isinstance(image_path1, tuple):
        image_path1, gt_fen = image_path1
    image1 = cv2.imread(image_path1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    white_turn = True
    try:
        board1, *_ = recognizer.predict(image1, parse_turn(image_path1))
    except ChessboardNotLocatedException as e:
        logger.debug("Chessboard not located in image {}\n".format(image_path1))
        image1_nobg = rembg.remove(image1)
        image1_nobg = cv2.cvtColor(image1_nobg, cv2.COLOR_RGBA2RGB)
        try:
            board1, *_ = recognizer.predict(image1_nobg, parse_turn(image_path1))
            image1 = image1_nobg
            logging.info("Chessboard successfully located after background subtraction in image {}\n".format(image_path1))
        except ChessboardNotLocatedException as e:
            logger.debug("Chessboard not located despite background subtraction in image {}\nConsider takin a better picture"
                         .format(image_path1))
            exit(1)
        # remove background and try again
    fens = []
    board1 = chess.Board()
    fens.append(board1.fen())

    for idx, image_path2 in enumerate(game['images'], 0):
        if idx == 0:
            continue
        if isinstance(image_path2, tuple):
            image_path2, gt_fen = image_path2
        logger.debug("Processing Images ({}-{}): {}-{}\n".format(idx-1, idx, image_path1, image_path2))

        # RUN INFERENCE
        image2 = cv2.imread(image_path2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        try:    
            board2, *_ = recognizer.predict(image2, parse_turn(image_path2))
        except ChessboardNotLocatedException as e:
            logger.debug("Chessboard not located in image {}\n".format(image_path2))
            image2_nobg = rembg.remove(image2)
            image2_nobg = cv2.cvtColor(image2_nobg, cv2.COLOR_RGBA2RGB)            
            try:
                board2, *_ = recognizer.predict(image2_nobg, parse_turn(image_path2))
                image2 = image2_nobg
                logging.info("Chessboard successfully located after background subtraction in image {}\n".format(image_path2))
            except ChessboardNotLocatedException as e:
                logger.debug("Chessboard not located despite background subtraction in image {}\nConsider takin a better picture"
                             .format(image_path2))
                exit(1)


        board2 = test_rejection(recognizer, board1, board2, image_path2)

        old_board = board2.copy()

        if board2 is None:
            logger.debug("Failed to converge to a solution for {}.\nExiting...".format(image_path2))
            raise ValueError("Failed to converge to a solution for {}.\nExiting...".format(image_path2))
        
        if board1.fen() == board2.fen():
            print('debug')

        diff_map, board1, board2 = polish_results(board1, board2, image_path1, image_path2, white_turn)
        
        fens.append(board2.fen())
        
        
        white_turn = not white_turn
        # SAVE & OUTPUT RESULTS
        final_frame = visualise_results(board1, old_board, board2, image1, image2, diff_map) 
        
        if not os.path.exists(os.path.join('results', game['name'])):
            os.makedirs(os.path.join('results', game['name']))
        cv2.imwrite('results/{}/{}-{}.png'.format(game['name'], idx-1, idx), final_frame)
        while image_queue.qsize() > MAX_QUEUE_LEN:
            time.sleep(0.1)

        image_queue.put(final_frame)
        
        image_path1 = image_path2
        image1 = image2
        board1 = board2

    time.sleep(0.01)
    
    with open('results/{}.txt'.format(game['name']), 'w') as f:
        f.write('\n'.join(fens))
    
    # TODO if PGN ground truth available compute a pgn and compare the results
    

def main(dataset):
    classifier_folder = URI("models://transfer_learning")

    for game in dataset:
        if game[0] == 'M':
            continue
        logger.info("Processing game (N. Images:{}): {}\n".format(len(dataset[game]['images']), dataset[game]))
        try:
            process_game(dataset[game], classifier_folder)
        except Exception as e:
            logger.error("Failed to process game {}\n".format(game))
            logger.error(e)
            print(e)
            failed_execution = True
            exit(1)


if __name__ == '__main__':
    # parse arguments TODO
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('main.log')
    file_handler.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdout_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    
    dataset = test_loader = get_dataloader(DATADIR, PGN_DIR)

    inference_thread = threading.Thread(target=main, args=(dataset,), daemon=True)
    inference_thread.start()

    display_images()
    
    inference_thread.join()