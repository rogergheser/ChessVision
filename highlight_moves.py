import chess
import chess.svg
import os
from utils import EmptySquare, board_to_matrix, matrix_to_fen, image_from_fen, PIL2cv2
from io_utils import svg_to_image
import cv2


FEN_ROOT = 'data/FENS/'
CHESSCOG_ROOT = './'
MY_WORK_ROOT = './'
RED = '#FF0000'
GREEN = '#00FF00'
ORANGE = '#FFA500'


def empty_to_piece(diff_map, board1, board2):
    """
    If a square from empty goes to a piece and no other square change accordingly,
    then the piece was placed on the board by mistake.
    """
    changes = [diff for diff in diff_map if diff[1] == 'empty' and diff[2] != 'empty']
    pass
## TODO FINISH THIS FUNCTION

    return changes, board1, board2

def same_color_diff_piece(diff_map, board1, board2):
    """
    Given a list of square and how they changed from one board to another,
    filter out any square on which a piece changed to another kind of the same color
    :param: diff_map: list - list of tuples (square, piece1, piece2)
    """
    # Filtering conditions
    condition1 = lambda square, piece1, piece2: piece1.islower() and piece2.islower() and piece1[0] != piece2[0]\
    and piece1 != 'empty' and piece2 != 'empty'
    condition2 = lambda square, piece1, piece2: piece1.isupper() and piece2.isupper() and piece1[0] != piece2[0]\
    and piece1 != 'empty' and piece2 != 'empty'

    try:
        changes = [diff for diff in diff_map if condition1(*diff) or condition2(*diff)]
    except:
        print("Issue")
    # if len(changes) > 0:
    #     print(f"Changes: {changes}")


    original_fen = board1.fen()
    map1, map2 = board_to_matrix(board1), board_to_matrix(board2)
    for square, piece1, piece2 in changes:
        i, j = square // 8, square % 8
        map2[i][j] = piece1
    new_fen = matrix_to_fen(map2)
    
    final_fen = new_fen.split(' ')[0] + ' ' + ' '.join(original_fen.split(' ')[1:])

    board2 = chess.Board(final_fen)
    board2.turn = not board1.turn


    return changes, board1, board2

def legal_move(board1, diff_map, white_turn):
    """
    :param: diff_map: list - list of tuples (square, piece1, piece2)
    """
    assert len(diff_map) == 2

    if white_turn:
        if diff_map[0][1].islower() and diff_map[1][2].islower():
            start_candidate = diff_map[0]
            end_candidate = diff_map[1]
        elif diff_map[0][2].islower() and diff_map[1][1].islower():
            start_candidate = diff_map[1]
            end_candidate = diff_map[0]
        else:
            raise ValueError("Unexpected state")
    else:
        if diff_map[0][1].isupper() and diff_map[1][2].isupper():
            start_candidate = diff_map[0]
            end_candidate = diff_map[1]
        elif diff_map[0][2].isupper() and diff_map[1][1].isupper():
            start_candidate = diff_map[1]
            end_candidate = diff_map[0]
        else:
            raise ValueError("Unexpected state")
        
    move = chess.Move(start_candidate[0], end_candidate[0])

    return move in board1.legal_moves


def last_move(fen1, fen2, white_turn):
    """
    Given two FEN strings, each indicating an inference state of a chess game, returns
    a list of candidate moves that could have been played to transition from the first
    state to the second state. The list is empty if the two states are not consecutive.
    :param fen1: str - FEN string of the first state
    :param fen2: str - FEN string of the second state
    :return: list - list of candidate moves
    """
    board1 = chess.Board(fen1)
    board2 = chess.Board(fen2)
    candidate_moves = board1.legal_moves
    
    # get differences between boards
    # diff = board1.piece_map().items() ^ board2.piece_map().items()

    diff_map = []
    board_map1, board_map2 = board1.piece_map(), board2.piece_map()
    for i in range(64):
        if i not in board_map1.keys():
            board_map1[i] = EmptySquare()
        if i not in board_map2.keys():
            board_map2[i] =  EmptySquare()
        
        if board_map1[i] != board_map2[i]:
            diff_map.append((i, board_map1[i].symbol(), board_map2[i].symbol()))

    constraints = [same_color_diff_piece]
    raw_total = len(diff_map)
    for constraint in constraints:
        changes, board1, board2 = constraint(diff_map, board1, board2)
        diff_map = [diff for diff in diff_map if diff not in changes]
    filtered_total = len(changes)

    print(f"Filtered out {raw_total - filtered_total} moves")
    print(diff_map)

    # if len(diff_map) == 2:
    #     # Here we only have squares that have changed
    #     # We can infer the move that was played
    #     if legal_move(board1, diff_map, white_turn):
    #         pass
    return diff_map, board1.fen(), board2.fen()

def get_move_arrow(square1, square2):
    """
    Given two squares, returns the move that was played to transition from the first square to the second square
    :param square1: (int, 'str', 'str') - square, piece1, piece2
    :param square2: (int, 'str', 'str') - square, piece1, piece2
    """
    if square1[2] == 'empty' and square1[1] != 'empty' \
        and square1[1] == square2[2]:
        return chess.svg.Arrow(square1[0], square2[0], color=ORANGE)
    elif square1[2] != 'empty' and square1[1] == 'empty' \
        and square1[2] == square2[1]:
        return chess.svg.Arrow(square2[0], square1[0], color=ORANGE)
    elif square1[2] == 'empty' and square2[2] != 'empty' and \
        square2[2] == square1[1]:
        # capture occured
        return chess.svg.Arrow(square1[0], square2[0], color=RED)
    elif square2[2] == 'empty' and square1[2] != 'empty' and \
        square1[2] == square2[1]:
        # capture occured
        return chess.svg.Arrow(square2[0], square1[0], color=RED)
        
def get_square_fill(changes):
    """
    Given a square and two pieces, returns the color to fill the square with
    """
    fill = {}
    for square, piece1, piece2 in changes:
        if piece1 == 'empty' and piece2 != 'empty':
            fill[square] = GREEN
        elif piece1 != 'empty' and piece2 == 'empty':
            fill[square] = GREEN
        elif piece1 != 'empty' and piece2 != 'empty' and piece1 != piece2:
            fill[square] = RED
    return fill


def highlight_changes(board, changes):
    """
    Given a chess.Board object and a list of changes, highlights the changes on the board
    """
    arrows = []
    if len(changes) == 2:
        arrows = [get_move_arrow(changes[0], changes[1])]
        if None in arrows:
            arrows = []
            fill = get_square_fill(changes)
        else:
            fill = {}
    else:
        fill = get_square_fill(changes)

    return fill, arrows
            

if __name__ == "__main__":
    with open(os.path.join(FEN_ROOT, 'Morphy-Dukes(Opera_Game)W-fens.txt'), 'r') as f:
        fens = f.readlines()
        fens = [fen.strip() for fen in fens]

    for i in range(len(fens) - 1):
        print(fens[i])
        print(fens[i+1])
        white_turn = i % 2 == 0
        unfiltered_fen2 = fens[i+1]
        changes, fen1, fen2 = last_move(fens[i], fens[i+1], white_turn)

        fill, arrows = highlight_changes(chess.Board(fen2), changes)

        print()
        fens[i] = fen1
        fens[i+1] = fen2

        # Display the PIL images on screen
        image1 = image_from_fen(fen1)
        image2 = image_from_fen(fen2)
        old_image2 = image_from_fen(unfiltered_fen2)
        final_res = chess.svg.board(chess.Board(fen2), arrows=arrows, fill=fill)
        final_res = svg_to_image(final_res)
        cv2.imshow('Image1', PIL2cv2(image1))
        cv2.imshow('Image2', PIL2cv2(image2))
        cv2.imshow('Old Image2', PIL2cv2(old_image2))
        cv2.imshow('Final Result', PIL2cv2(final_res))
        cv2.moveWindow('Image1', 0, 0)
        cv2.moveWindow('Image2', 400, 0)
        cv2.moveWindow('Final Result', 0, 400)
        cv2.moveWindow('Old Image2', 400, 400)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(last_move(fen1, fen2))