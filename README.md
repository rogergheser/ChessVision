# ChessVision
This project combines the efforts of [chesscog](https://github.com/georg-wolflein/chesscog) by Georg Wölflein and some of my own work which consists of fine-tuning the model, creting a small dataset and adding some chess game logic to remove inconsistencies in the board state.

This project is still in progress. The final objectives is to parse the full game from video, but the current state of the project is to parse a game from a sequence of images taken by a user. A whole pipeline to select the video frames and process them in real time is still to be implemented.

# Overview
The model consists of a pretrained occupancy classifier (ResNet) and piece classifier (InceptionV3). The model was trained by the authors of chesscog on a dataset of ~5,000 synthetically generated images (3D renderings of chess positions from different angles and with varying light).
After investigating the limits of this approach I shot pictures of 3 chess games, from both player perspectives, move-by-move. The dataset consists of 358 images, 179 for each perspective. 
Some additional images were taken to test the model on a different chess boards and chess sets than the one used in the training set.


# Installation
```bash
pip install -r requirements.txt
pip install cairosvg
```

If you run into installation issues with cairosvg and avoit issues run
```bash
conda install cairo pango gdk-pixbuf libffi cairosvg
```

# ChessVision
The purpose of this project is to parse a complete chess game from a video.
To tackle this problem I addressed the simpler issue of chess position recognition by exploiting a pre-trained model by Georg Wölflein et al. and fine tuning.

The repo suggests taking a picture of the starting position from both sides. I deemed the results to be unsatisfactory, thus repeating the process with a larger pool of images. To ease the process of obtaining labels (FENs) for positions, I took pictures of chess games and parse the FEN directly from the PGN.

The dataset will be made publicly available, it consists of pictures taken from both perspectives (white and black) move-by-move of famous chess games, including:
* Morphy's Opera Game
* Alekhine - Nimzowitsch (1930)
* Tal - Hjartarson 1987


# Usage
For sheer training, evaluating and testing or corner detection the model please refer to the [chesscog](https://github.com/georg-wolflein/chesscog) repository. 
To run the inference process on a sequence of images, run main.py with the path to the folder containing the images as an argument.
```bash
python main.py path/to/folder
```

# Evaluation
An automatic evaluation and benchmarking process is yet to be implemented.

# Parsing from video
The final goal of this project is to parse a chess game from a video. This work is still in progress and yet to be devoloped.

# Inference Results
The inference process is still faulty, but the occupancy classifier mostly detects all pieces and rarely includes false negatives. This means that illogical board states can be parsed out.

<img src='source/confusion_matrix.png' width=80%>

This implementation can filter out the following:
* Piece moving from one square to another, but on the second board state it is misclassified.
* Pawns appearing on the first or last rank are filtered out.
* Pieces appearing on squares that are unreachable via legal moves are removed.



<img src='source/opera_game.gif' width=50%>


[def]: source/opera_game.gif