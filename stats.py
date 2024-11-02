import os
import pandas as pd
import chess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chess.pgn
class Metrics():
    def __init__(self):
        self.CATEGORIES = [
            "P", "N", "B", "R", "Q", "K",
            "p", "n", "b", "r", "q", "k",
            "empty"
        ]
        # The following hold predictions and ground truths for each square
        self.parsed_board = []
        self.predictions = []
        self.ground_truths = []
        self.precisions = []
        self.ranked_precisions = []
        self.conf_matrix = None
        self.correct_pos = 0
        self.total_pos = 0
        self.board_accuracies = []
    
    def create_report(self, path="results/"):
        """
        Create a report of the metrics.
        """
        with open(os.path.join(path, 'report.txt'), "w") as f:
            f.write("Occupancy stats: {}\n".format(self.occupancy_stats()))
            f.write("Piece stats: {}\n".format(self.piece_stats()))
            f.write("mAP: {}\n".format(self.mAP()))
            f.write("Correct positions: {}\n".format(self.correct_pos/self.total_pos))
            f.write("Board accuracy: {}\n".format(self.board_accuracy()))
            f.write("Recognised boards: {}\n".format(self.recognised_boards()))
            self.confusion_matrix()
            self.draw_confusion_matrix("metrics", resdir=path)

    def transfer_values(self, metrics):
        """
        Add values from another metrics object
        """
        self.predictions += metrics.predictions
        self.ground_truths += metrics.ground_truths
        self.precisions += metrics.precisions
        self.ranked_precisions += metrics.ranked_precisions
        self.correct_pos += metrics.correct_pos
        self.total_pos += metrics.total_pos
        self.board_accuracies += metrics.board_accuracies
        self.parsed_board += metrics.parsed_board
        

    def reset(self):
        self.predictions = []
        self.ground_truths = []
        self.precisions = []
        self.ranked_precisions = []
        self.conf_matrix = None
        self.correct_pos = 0
        self.total_pos = 0
        self.board_accuracies = []
        self.parsed_board = []

    def update_board(self, is_recognised:bool):
        self.parsed_board.append(1 if is_recognised else 0)
        if not is_recognised:
            self.board_accuracies.append(0)

    def update(self, ground_truth_fen, predicted_fen):
        """
        Update the metrics with the ground truth and predicted FENs.
        """
        self.update_board(True)

        # ADD Equal FENs check

        ground_truth_board = chess.Board(ground_truth_fen)
        predicted_board = chess.Board(predicted_fen)
        precisions = []

        for i in range(64):
            gt_piece = ground_truth_board.piece_at(i)
            pred_piece = predicted_board.piece_at(i)

            if gt_piece is not None:
                self.ground_truths.append(gt_piece.symbol())
            else:
                self.ground_truths.append('empty')

            if pred_piece is not None:
                self.predictions.append(pred_piece.symbol())
            else:
                self.predictions.append('empty')

            # Calculate precision for this square
            if pred_piece is not None and gt_piece is not None:
                if pred_piece.symbol() == gt_piece.symbol():
                    self.precisions.append(1)
                else:
                    self.precisions.append(0)
            elif pred_piece is None and gt_piece is None:
                self.precisions.append(1)
            else:
                self.precisions.append(0)

        self.ranked_precisions.append( sum(self.precisions[-64:])/64 )
        if self.ranked_precisions[-1] == 1.0:
            self.correct_pos += 1
        self.total_pos += 1

    def board_accuracy(self):
        return sum(self.board_accuracies)/len(self.board_accuracies)

    def recognised_boards(self):
        return sum(self.parsed_board)/len(self.parsed_board)            

    def occupancy_stats(self):
        """
        Returns accuracy, recall, precision and f1-score of occupancy classification.
        If a piece is on the square it counts as positive, if the square is empty it counts as negative.
        """
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for pred, gt in zip(self.predictions, self.ground_truths):
            if pred not in self.CATEGORIES or gt not in self.CATEGORIES:
                raise ValueError("Unexpected behaviour")
            
            if pred != 'empty':
                if gt != 'empty':
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if gt != 'empty':
                    false_negatives += 1
                else:
                    true_negatives += 1

        accuracy = (true_negatives + true_positives) / (true_negatives + false_positives + true_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall)/ (precision + recall)

        return {'accuracy': accuracy, 'precision':precision, 'recall':recall, 'f1':f1}

    def piece_stats(self):
        """
        Returns macro-average accuracy, recall, precision and f1-score of piece classification.
        """
        hit = 0
        miss = 0

        for pred, gt in zip(self.predictions, self.ground_truths):
            if pred == gt:
                hit += 1
            else:
                miss += 1

        accuracy = hit / (hit + miss)

        for pred, gt in zip(self.predictions[-64:], self.ground_truths[-64:]):
            if pred == gt:
                hit += 1
            else:
                miss += 1
        
        self.board_accuracies.append(1 if hit/(hit+miss) == 1.0 else 0)

        return accuracy
    
    def confusion_matrix(self):
        """
        Returns a confusion matrix for the piece classification.
        """
        matrix = np.zeros((len(self.CATEGORIES), len(self.CATEGORIES)), dtype=np.int32)

        for pred, gt in zip(self.predictions, self.ground_truths):
            matrix[self.CATEGORIES.index(gt), self.CATEGORIES.index(pred)] += 1

        return matrix

    def draw_confusion_matrix(self, name, resdir='results/plots/'):
        if self.conf_matrix is None:
            self.conf_matrix = self.confusion_matrix()
        
        row_sums = self.conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix_normalized = self.conf_matrix / row_sums

        fig, ax = plt.subplots()
        ax = sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True)

        # Add axis labels
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        
        ax.set_xticks(np.arange(len(self.CATEGORIES)) + 0.5)
        ax.set_yticks(np.arange(len(self.CATEGORIES)) + 0.5)

        ax.set_xticklabels(self.CATEGORIES, rotation=45, ha="right")
        ax.set_yticklabels(self.CATEGORIES, rotation=0)
        ax.set_title('Confusion Matrix - {} Values'.format(sum(row_sums)))
        # Show the plot
        plt.tight_layout()
        plt.savefig('{}_conf.png'.format(os.path.join(resdir, name)))
    
    def mAP(self):
        """
        Returns the mean average precision of the occupancy classification.
        """
        return sum(self.ranked_precisions) / len(self.ranked_precisions)

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
    FEN_DIR = 'results'
    PGN_DIR = 'data/PGNS/chess_games'

    for game in os.listdir(PGN_DIR):
        fen_file = 'FEN-{}'.format(game)
        fen_file = fen_file.replace('.pgn', '.txt')

        if not os.path.exists(os.path.join(FEN_DIR, fen_file)):
            print("Skipping game cause no predicted fen was found")
            print(os.path.join(FEN_DIR, fen_file))
            continue

        fen_file = os.path.join(FEN_DIR, fen_file)
        pgn_file = os.path.join(PGN_DIR, game)

        with open(fen_file, "r") as f:
            predicted_fens = f.readlines()

        with open(pgn_file, "r") as f:
            pgn = f.readlines()

        metrics = Metrics()
        pgn = open(pgn_file)
        game = chess.pgn.read_game(pgn)
        board = chess.Board()

        for idx, (pred_fen, gt_move) in enumerate(zip(predicted_fens, game.mainline_moves())):
            gt_fen = board.fen()
            pred_fen = pred_fen.strip()

            metrics.update(gt_fen, pred_fen)
            print("[Move {}] mAP: {}".format(idx, metrics.mAP()))
            board.push(gt_move)

        print("Occupancy stats: {}".format(metrics.occupancy_stats()))
        print("Piece stats: {}".format(metrics.piece_stats()))
        print("mAP: {}".format(metrics.mAP()))
        print("Correct positions: {}".format(metrics.correct_pos/metrics.total_pos))