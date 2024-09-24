from chesscog.chesscog.corner_detection import find_corners
import os
import json

def process_corners(dir):
    for file in os.listdir(dir):
        if file == '.DS_Store':
            continue
        if file.endswith('.json'):
            name = file.split('.')[0]
            path = os.path.join(dir, name, '.png')
            corners = find_corners(path)

if __name__ == '__main__':
    dir = 'Users/amirgheser/chess_data/transfer_learning/images/test'
    process_corners(dir)
