####
# Given a directory of images taken in sequence renames them appropriately, generates corner predictions and saves them in a directory
# Does not handle storing of pgn files
import argparse
import os
import rembg
import shutil
import re
from tqdm import tqdm

REMBG = False
INT_PATTERN = 'IMG_{}.jpeg'
rename_format = '{}{}.png'

def parse_int(s):
    if s == '.DS_Store':
        return -1
    return int(re.search(r'\d+', s).group())
    
def create_dataset(source_dir, dest_dir):

    for subdir in os.listdir(source_dir):
        if subdir.startswith('.'):
            continue
        dest_subdir = os.path.join(dest_dir, subdir)
        os.makedirs(dest_subdir, exist_ok=True)
        color = subdir[-1]
        subdir_path = os.path.join(source_dir, subdir)
        skipped = 0

        if os.path.isdir(subdir_path):
            print(f'Processing {subdir}')
            loop = tqdm(enumerate(sorted(os.listdir(subdir_path), key=lambda x: parse_int(x))), desc="Removing bg", total=len(os.listdir(subdir_path)))
            for i, img in loop:
                if img.startswith('.'):
                    skipped += 1
                    continue
                img_path = os.path.join(subdir_path, img)
                dest_path = os.path.join(dest_subdir, rename_format.format(i-skipped, color))
                if REMBG:
                    with open(img_path, 'rb') as f:
                        input_img = f.read()
                    output_img = rembg.remove(input_img)
                    with open(dest_path, 'wb') as f:
                        f.write(output_img)
                else:
                    shutil.copyfile(img_path, dest_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset from a directory of images')
    parser.add_argument('source_dir', type=str, help='Directory containing images')
    parser.add_argument('dest_dir', type=str, help='Directory to save images to')
    parser.add_argument('--rembg', action='store_true', default=False, help='Remove background from images')

    args = parser.parse_args()

    source_dir = args.source_dir
    dest_dir = args.dest_dir
    REMBG = args.rembg
    
    create_dataset(source_dir, dest_dir)
