import os
import chess.pgn
import json 
import argparse
import cv2
import numpy as np
import sys
from recap import URI, CfgNode as CN
from chesscog.corner_detection import find_corners
def fens_from_pgn(pgn_file, out_dir=None):
    """Extract FENs from a PGN file and save them to a directory.
    
    Args:
        png_file (str): The path to the PNG file.
        out_dir (str): The path to the output directory.
    """
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    
    pgn = open(pgn_file)
    game = chess.pgn.read_game(pgn)

    fens = []
    board = game.board()
    fens.append(board.fen())
    for idx, move in enumerate(game.mainline_moves()):
        board.push(move)
        fen = board.fen()
        fens.append(fen)
        if out_dir:
            with open(os.path.join(out_dir, f'move{idx}.txt'), 'a') as f:
                f.write(fen + '\n')
                f.flush()
    return fens

def get_corners(img_file):
    cfg = CN.load_yaml_with_base("config://corner_detection.yaml")
    filename = URI(img_file)
    img = cv2.imread(str(filename))
    corners = find_corners(cfg, img)

    return corners

def labels_from_pgn(pgn_file, out_dir, white_turn=True):
    """Extract labels from a PGN file and save them to a directory.
    
    Args:
        pgn_file (str): The path to the PGN file.
        out_dir (str): The path to the output directory.
        color (str): The color of the player.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    pgn = open(pgn_file)
    game = chess.pgn.read_game(pgn)
    if game is None:
        raise ValueError('No game found in PGN file.')
    
    board = game.board()

    img_file = os.path.join(out_dir, '0{}.png'.format('W' if white_turn else 'B'))
    corners = get_corners(img_file)

    data = {
        "white_turn": white_turn,
        "fen": board.fen(),
        "corners": np.matrix.tolist(corners)
        }
    
    img_name = os.path.join(out_dir, '0{}.json'.format('W' if white_turn else 'B'))
    with open(img_name, 'w') as f:
        json.dump(data, f)
        f.flush()

    for idx, move in enumerate(game.mainline_moves()):
        board.push(move)
        
        img_file = os.path.join(out_dir, '{}{}.png'.format(idx+1, 'W' if white_turn else 'B'))
        try:
            corners = get_corners(img_file)
        except Exception as e:
            print(f'Failed to find corners for {img_file}')
            print(e)
            continue
        data = {
            "white_turn": white_turn,
            "fen": board.fen(),
            "corners": np.matrix.tolist(corners)
            }

        img_name = os.path.join(out_dir, '{}{}.json'.format(idx+1, 'W' if white_turn else 'B'))
        with open(img_name, 'w') as f:
            json.dump(data, f)
            f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    pgn_file = '../morphy.pgn'
    out_dir = '/Users/amirgheser/chess_data/transfer_learning/images/test'
    white_turn = True
    
    args = parser.parse_args()
    labels_from_pgn(pgn_file, out_dir, white_turn)