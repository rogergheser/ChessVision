import rembg
import cv2
import os

src = 'data/chess_games/Alekhine-Nimzowitsch(1930)B/0B.jpeg'
out = 'test.png'

with open(src, "rb") as input_file:
    input_data = input_file.read()
    output_data = rembg.remove(input_data)

with open(out, "wb") as output_file:
    output_file.write(output_data)