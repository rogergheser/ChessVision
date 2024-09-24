import os
import shutil
import argparse

def parse_for_sorting(file):
    try:
        ret = int(file.split('.')[0][:-1])
    except ValueError:
        if file == '.DS_Store':
            return -1
        else:
            raise ValueError(f'Failed to parse {file}')
    return ret

def rename(dir):
    idx = 0 
    for file in sorted(os.listdir(dir), key=lambda x: parse_for_sorting(x)):
        if file == '.DS_Store':
            continue
        if file.endswith('W.jpeg'):
            path = os.path.join(dir, file)
            # color = 'W' if dir[-1] == 'W' else 'B'
            color = 'W'
            new_path = os.path.join(dir, f'{idx}{color}.png')
            shutil.move(path, new_path)
            idx+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='Directory containing images to rename')
    args = parser.parse_args()
    if not os.path.exists(args.dir):
        raise FileNotFoundError(f'Directory {args.dir} not found')
    rename(args.dir)