import chess
import chess.svg
from io_utils import svg_to_image
import cv2
import numpy as np

class EmptySquare:
    def __init__(self):
        pass
    
    def symbol(self):
        return 'empty'

    def __eq__(self, other):
        return isinstance(other, EmptySquare)

def board_to_matrix(board):
    """
    Given a chess.Board object, returns a matrix representation of the board
    Piece are classified as follows:
    [K, Q, R, B, N, P] for white pieces
    [k, q, r, b, n, p] for black pieces
    EmptySquare for empty squares
    """
    matrix = [[EmptySquare() for _ in range(8)] for _ in range(8)]
    for i in range(8)[::-1]: # chess.Board mapping is reversed, with white at the bottom
        for j in range(8):
            piece = board.piece_at(i * 8 + j)
            if piece is not None:
                matrix[i][j] = piece.symbol()
    return matrix

def image_from_fen(fen):
    """
    Given a FEN string, returns a PIL image of the board
    """
    board = chess.Board(fen)
    svg = chess.svg.board(board=board)
    return svg_to_image(svg)

def PIL2cv2(PIL_img):
    """
    Convert a PIL image to a cv2 image
    """
    return cv2.cvtColor(np.array(PIL_img), cv2.COLOR_RGB2BGR)

def matrix_to_fen(board):
    """
    Given a matrix representation of a chess board, returns the FEN string
    Piece are classified as follows:
    [K, Q, R, B, N, P] for white pieces
    [k, q, r, b, n, p] for black pieces
    EmptySquare for empty squares
    """
    fen = ''
    for i in range(8)[::-1]:
        empty = 0
        for j in range(8):
            piece = board[i][j]
            if isinstance(piece, EmptySquare):
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += piece
        if empty > 0:
            fen += str(empty)
        if i > 0:
            fen += '/'
    return fen
