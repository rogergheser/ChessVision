import os
import sys
import chess
import chess.svg
import cv2
import numpy as np
from PIL import Image
import cairosvg
import io
from chesscog.chesscog.recognition.recognition import ChessRecognizer
from chesscog.chesscog.occupancy_classifier.download_model import ensure_model as ensure_occupancy_classifier
from chesscog.chesscog.piece_classifier.download_model import ensure_model as ensure_piece_classifier
from recap import URI
from pathlib import Path
from chess import Status

GAMES_DIR = 'data/chess_games'
# GAMES_DIR = 'data/new_chess_games'

test_dir = os.path.join(GAMES_DIR, 'Morphy-Dukes(Opera_Game)W')
res_dir = os.path.join(GAMES_DIR, 'Morphy-Dukes(Opera_Game)W')

class GameMetrics():
    def __init__(self):
        pass

class Metrics():
    def __init__(self, board, true_fen):
        self.total = 0
        self.correct = 0
        self.incorrect = 0
        self.illegal = 0
        self.board = board
        self.true_fen = true_fen

    def compute_stats(self):
        for i in range(0, 64):
            if self.board.piece_at(i) == self.true_fen.piece_at(i):
                self.correct += 1
            else:
                self.incorrect += 1


def svg_to_image(svg_data):
    # Create a blank image
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(io.BytesIO(png_data))
    
    return image

def parse_file(file):
    """
    Parse file to extract index for sorting
    """
    return int(file.split('.')[0][:-1])

def main(file, white, classifiers_folder: Path = URI("models://"), setup: callable = lambda: None):
    """Main method for running inference from the command line.

    Args:
        classifiers_folder (Path, optional): the path to the classifiers (supplying a different path is especially useful because the transfer learning classifiers are located at ``models://transfer_learning``). Defaults to ``models://``.
        setup (callable, optional): An optional setup function to be called after the CLI argument parser has been setup. Defaults to lambda:None.
    """

    setup()

    img = cv2.imread(str(URI(file)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    recognizer = ChessRecognizer(classifiers_folder)
    board, *_ = recognizer.predict(img, white)

    print(board)
    print()
    print(
        f"You can view this position at https://lichess.org/editor/{board.board_fen()}")

    if board.status() != Status.VALID:
        print()
        print("WARNING: The predicted chess position is not legal according to the rules of chess.")
        print("         You might want to try again with another picture.")

    return board


if __name__ == "__main__":
    
    if not os.path.exists(test_dir):
        print(f'Invalid test directory: {test_dir}')
        sys.exit(1)

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    rets = []
    failed = []
    fens = []
    for file in os.listdir(test_dir):
        if file == '.DS_Store':
            continue
        path = os.path.join(test_dir, file)
        color_char = file.rsplit(".")[0][-1]
        if color_char == 'W':
            white = chess.WHITE
        elif color_char == 'B':
            white = chess.BLACK
        else:
            failed.append(file)
            print(f'Invalid file name: {file}')
            continue

        try:    
            board = main(file=path, white=white,
                         classifiers_folder=URI("models://transfer_learning"),
                          setup=lambda: [ensure_model(show_size=True)
                            for ensure_model in (ensure_occupancy_classifier, ensure_piece_classifier)])
        except Exception as e:
            print(f'Failed to process {file}')
            print(e)
            continue
        rets.append((file, board))
    print(f'\n\nParsed {len(rets)} games correctly')
    print(f'Failed to process {len(failed)} games')
    print(failed)

    for ret in sorted(rets, key=lambda x: parse_file(x[0])):
        game_name = ret[0].split('.')[0]
        fens.append(ret[1].board_fen())
        
        image1 = cv2.imread(os.path.join(test_dir, ret[0]))
        svg_data = chess.svg.board(ret[1], orientation=ret[1].turn)
        image2 = svg_to_image(svg_data)
        # resize the image so that the heights match and the ratio is preserved
        image2 = image2.resize((image1.shape[1], (image2.width * image1.shape[1] // image2.height)))
        image2 = np.array(image2)
        # stitch the images together
        image = np.concatenate((image1, image2), axis=0)
        cv2.imwrite(os.path.join(res_dir, f'{game_name}.png'), image)

    with open('../my_work/FENS/{}-fens.txt'.format(test_dir.split('/')[-1]), 'w') as f:
        for fen in fens:
            f.write(fen + '\n')
    
    print(f'\n\nSaved {len(rets)} games correctly')
    
