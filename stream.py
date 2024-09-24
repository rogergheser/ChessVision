import cv2
import time
import argparse
import threading
import queue
import rembg
import logging
import PIL
import chess
import chess.svg
from chesscog.chesscog.core.exceptions import ChessboardNotLocatedException
from chesscog.chesscog.recognition import ChessRecognizer
from recap import URI
from PIL import Image
from main import acceptable_diff, add_noise, polish_results, get_diff_map, get_square_fill, create_final_frame
# Global variables
FPS = 2
DELAY = int(1000/FPS)
RUNNING = True
MAX_RETRIES = 5
SAVE_RESULTS = False
BOARD_SIZE = 400
Frames = queue.Queue()
Fens = queue.Queue()
Image_queue = queue.Queue()

def visualise_results(board1, old_board2, board2, image1, image2, diff_map):
    """
    :param: board1: chess.Board - image of the board state at time t
    :param: old_board2: chess.Board - image of the board state at time t+1 before being polished
    :param: board2: chess.Board - image of the board state at time t+1
    :param: image1: PIL.Image - image of the board state at time t
    :param: image2: PIL.Image - image of the board state at time t+1
    :param: diff_map: list - list of tuples containing the square, piece and state of the square
    :return: move_frame, final_frame: PIL.Image, PIL.Image - image of the board with the move highlighted and the final frame
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

    return move_frame, final_frame

def test_rejection(recognizer, board1, board2, frame2, white_perspective):
    image2 = PIL.Image.fromarray(frame2)
    patience = MAX_RETRIES

    while patience > 0 and not acceptable_diff(board1, board2):
        logger.debug("Retrying inference with extra noise and patience:{}\n".format(patience))
        
        # add some random noise to the image
        image2 = add_noise(image2, patience)
        board2, *_ = recognizer.predict(image2, white_perspective)
        patience -= 1

    if patience == 0:
        logger.debug('Failed to converge to a solution for {}.\nPlease change the image or fine tune the model better.\nExiting...')
        raise ValueError('Failed to converge to a solution for {}.\nPlease fine tune the model better.\nIgnoring frame...')
    logger.debug('Successfully converged to a solution after {} tries'.format(MAX_RETRIES-patience))

    return board2

def parse_move(recognizer, white_perspective, board1, frame2, white_turn):
    """
    This function takes two consecutive frames, detects changes and 
    if there are enough changes it returns a new board else, returns no changes were made
    """
    ret = True
    frame2 = cv2.cvtColor(rembg.remove(frame2), cv2.COLOR_BGRA2RGB)

    # Get the board from the second frame
    try:
        board2, *_ = recognizer.predict(frame2, white_perspective)
    except:
        logger.error('Error detecting board')
        raise ChessboardNotLocatedException('Error detecting board')

    board2 = test_rejection(recognizer, board1, board2, frame2)    
    old_board = board2.copy()

    if board2 is None:
        logger.debug('No board detected. Skipping frame')
        raise ValueError('Not handled this case yet')
    
    if board1.fen() == board2.fen():
        raise ValueError('No changes were made. Skipping frame')
    
    diff_map, board1, board2 = polish_results(board1, board2)
    
    Fens.put(board2.fen())
    
    
    if SAVE_RESULTS:
        raise('Not implemented yet')

    # TODO move outside of this function in outer loop
    board1 = board2
    white_turn = not white_turn
    
    return ret, board1, old_board, board2, diff_map


def main(cap, white_perspective, delay=DELAY, classifier_folder=URI("models://transfer_learning")):
    move_counter = 0
    recognizer = ChessRecognizer(classifier_folder)

    running = True
    ret, frame1 = cap.read()
    if not ret:
        print('Error reading video')
        RUNNING = False
        return
    
    frame1 = cv2.cvtColor(rembg.remove(frame1), cv2.COLOR_BGRA2RGB)
    Frames.put(frame1)
    try:
        board1, *_ = recognizer.predict(frame1, white_perspective)
    except:
        logger.error('Error detecting board')
        raise ChessboardNotLocatedException('Error detecting board')

    time.sleep(delay/1000)
    white_turn = True

    while RUNNING:
        ret, frame2 = cap.read()
        if not ret:
            print('Error reading video')
            RUNNING = False
            break
        Frames.put(frame2)

        try:
            ret, board1, old_board, board2, diff_map = parse_move(recognizer, white_perspective, board1, frame2, white_turn)
            if not ret:
                logger.debug('No move detected')
                raise ValueError('No move detected')
            logger.debug('Move {} detected'.format(move_counter))
            move_counter += 1
            move_frame, final_frame = visualise_results(board1, old_board, board2, frame1, frame2, diff_map)
            Image_queue.put(move_frame)

        except ValueError as e:
            raise('Unhandled exceptions')
            logger.debug(e)
            continue


        # concatenate the two frames

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            RUNNING = False

        frame1 = frame2
        board1 = board2
        white_turn = not white_turn
    cap.release()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--fps', type=int, help='Frames per second')
    args.add_argument('--video', type=str, help='Video file')
    args.add_argument('--camera', type=int, help='Camera number (default 0)')
    args.add_argument('--white_perspective', type=bool, help='White perspective (default True)')
    args = args.parse_args()

    if args.fps:
        FPS = args.fps
        DELAY = int(1000/FPS)

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)

    white_perspective = True
    if args.white_perspective:
        white_perspective = args.white_perspective

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

    main_thread = threading.Thread(target=main, args=(cap, white_perspective, DELAY))