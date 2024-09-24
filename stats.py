import pandas as pd
import chess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Stats():
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = self.parse()
        self.fen_actual = self.df['fen_actual']
        self.fen_predicted = self.df['fen_predicted']
        self.CATEGORIES = [
            "P", "N", "B", "R", "Q", "K",
            "p", "n", "b", "r", "q", "k",
            "empty"
        ]
        confusion_matrix = self.confusion_matrix()
        self.draw_confusion_matrix(confusion_matrix)

        # Compute:
        # Piece Confusion matrix
        # Free square statistics
        
    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        # create a confusione matrix for classification considering the classes:
        # P, N, B, R, Q, K, p, n, b, r, q, k, empty
        matrix = np.zeros((len(self.CATEGORIES), len(self.CATEGORIES)), dtype=np.int32)

        # create list of values to be passed to confusion matrix function
        y_true = []
        y_pred = []

        for i in range(len(self.fen_actual)):
            true_board = chess.Board(self.fen_actual[i])
            pred_board = chess.Board(self.fen_predicted[i])
            for i in range(64):
                true_piece = true_board.piece_at(i)
                pred_piece = pred_board.piece_at(i)
                if true_piece is not None:
                    y_true.append(true_piece.symbol())
                else:
                    y_true.append('empty')
                if pred_piece is not None:
                    y_pred.append(pred_piece.symbol())
                else:
                    y_pred.append('empty')

        matrix = confusion_matrix(y_true, y_pred, labels=self.CATEGORIES)

        return matrix
    
    def draw_confusion_matrix(self, conf_matrix):
        class_labels = [
    "P", "N", "B", "R", "Q", "K",
    "p", "n", "b", "r", "q", "k",
    "empty"
]
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix_normalized = conf_matrix / row_sums

        fig, ax = plt.subplots()
        ax = sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True)

        # Add axis labels
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        
        ax.set_xticks(np.arange(len(class_labels)) + 0.5)
        ax.set_yticks(np.arange(len(class_labels)) + 0.5)

        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticklabels(class_labels, rotation=0)

        # Show the plot
        plt.tight_layout()
        plt.savefig('chesscog/results/transfer_learning/recognition/confusion_matrix.png')        

    def parse(self):
        df = pd.read_csv(self.csv_file)
        return df
    

if __name__ == '__main__':
    path = 'chesscog/results/transfer_learning/recognition/test.csv'
    stats = Stats(path)
    df = stats.parse()
    print(df)
            