import re
import os
import random
import shutil

src_dir = '/Users/amirgheser/chess_data/transfer_learning/images/test'
dst_dir = '/Users/amirgheser/chess_data/transfer_learning/images/train'

def highest_num(dir):
    nums = []
    for file in os.listdir(dir):
        num = re.search(r'\d+', file)
        if num:
            nums.append(int(num.group()))
    return max(nums)

def move_files():
    N = highest_num(src_dir)
    values = random.sample(range(N+1), int(.8*N))

    print(values)

    for num in values:
        try:
            shutil.move(f'{src_dir}/{num}W.png', f'{dst_dir}/{num}W.png')
            shutil.move(f'{src_dir}/{num}B.json', f'{dst_dir}/{num}B.json')
        except:
            continue
        try:
            shutil.move(f'{src_dir}/{num}B.png', f'{dst_dir}/{num}B.png')
            shutil.move(f'{src_dir}/{num}W.json', f'{dst_dir}/{num}W.json')
        except:
            continue
if __name__ == '__main__':
    move_files()