import glob
import os

def rename(dir, color):
    for idx, filename in enumerate(os.listdir(dir)):
        old_name = os.path.join(dir, filename)
        new_name = os.path.join(dir, str(idx) + color + '.png')
        os.rename(old_name, new_name)

if __name__ == '__main__':
    color = 'B'
    dir = 'source/new_chess_games/black-view'
    rename(dir, color)
        