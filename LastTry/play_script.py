import chess
import chess.engine
import random
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, conv_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv_size, out_channels=conv_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_size)
        self.conv2 = nn.Conv2d(in_channels=conv_size, out_channels=conv_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_size)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ResidualChessModel(nn.Module):
    def __init__(self, conv_size, conv_depth):
        super(ResidualChessModel, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels=14, out_channels=conv_size, kernel_size=3, padding=1)
        self.residual_layers = nn.Sequential(*[ResidualBlock(conv_size) for _ in range(conv_depth)])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 8 * conv_size, 1)

    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        x = self.residual_layers(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.fc(x))
        return x

class ChessConvNet(nn.Module):
    def __init__(self, conv_size, conv_depth):
        super(ChessConvNet, self).__init__()
        # Define the convolutional layers
        self.convs = nn.Sequential(
            *[
                nn.Conv2d(in_channels=conv_size if i > 0 else 1, 
                          out_channels=conv_size, 
                          kernel_size=3, 
                          padding='same') for i in range(conv_depth)
            ]
        )
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(conv_size * 14 * 8 * 8, 64)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x
        
model = ResidualChessModel(conv_size=16, conv_depth=6)
model.load_state_dict(torch.load('model_weights.pth'))
model  = model.to('cuda')
squares_index = {
    'a' : 0,
    'b' : 1,
    'c' : 2,
    'd' : 3,
    'e' : 4,
    'f' : 5,
    'g' : 6,
    'h' : 7
}

def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]

def split_dims(board):
    board3d = np.zeros((14,8,8))
    
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1
        
    aux = board.turn
    
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i,j = square_to_index(move.to_square)
        board3d[12][i][j]
        
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i,j = square_to_index(move.to_square)
        board3d[13][i][j]
    
    board.turn = aux
    
    return board3d
import chess.svg
from IPython.display import display, clear_output
import time
import chess.pgn
import random

class TreeNode:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

def mcts_get_move(board, simulations=100):
    root = TreeNode()

    for _ in range(simulations):
        node = root
        temp_board = board.copy()

        # Selection phase
        while node.children:
            node = max(node.children, key=lambda n: (n.wins / (n.visits or 1)) + (2 * (node.visits ** 0.5) / (n.visits or 1)) ** 0.5)
            temp_board.push(node.move)

        # Expansion phase
        legal_moves = list(temp_board.legal_moves)
        if legal_moves:
            selected_move = random.choice(legal_moves)
            temp_board.push(selected_move)
            new_node = TreeNode(move=selected_move, parent=node)
            node.children.append(new_node)
            node = new_node

        # Simulation phase
        while not temp_board.is_game_over():
            random_move = random.choice(list(temp_board.legal_moves))
            temp_board.push(random_move)

        # Backpropagation phase
        result = temp_board.result()
        while node is not None:
            node.visits += 1
            if result == '1-0' and temp_board.turn == chess.WHITE:
                node.wins += 1
            elif result == '0-1' and temp_board.turn == chess.BLACK:
                node.wins += 1
            node = node.parent

    best_node = max(root.children, key=lambda n: n.visits)
    return best_node.move if best_node else random.choice(list(board.legal_moves))
def play_and_save_game(engine_path, game_number, size=600, wait_time=0.5):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        while True:
            move = mcts_get_move(board, simulations=7000)
            board.push(move)
            node = node.add_variation(move)

            clear_output(wait=True)
            display(chess.svg.board(board=board, size=size))
            time.sleep(wait_time)

            if board.is_game_over():
                if board.result() == '1-0':
                    print('Checkmate! White (AI) wins!')
                    game.headers["Result"] = '1-0'
                elif board.result() == '0-1':
                    print('Checkmate! Black (Stockfish) wins!')
                    game.headers["Result"] = '0-1'
                elif board.result() == '1/2-1/2':
                    print('Draw!')
                    game.headers["Result"] = '1/2-1/2'
                break

            result = engine.play(board, chess.engine.Limit(depth=2))
            board.push(result.move)
            node = node.add_variation(result.move)

            clear_output(wait=True)
            display(chess.svg.board(board=board, size=size))
            time.sleep(wait_time)

            if board.is_game_over():
                if board.result() == '1-0':
                    print('Checkmate! White (AI) wins!')
                    game.headers["Result"] = '1-0'
                elif board.result() == '0-1':
                    print('Checkmate! Black (Stockfish) wins!')
                    game.headers["Result"] = '0-1'
                elif board.result() == '1/2-1/2':
                    print('Draw!')
                    game.headers["Result"] = '1/2-1/2'
                break

    # Save the game to a PGN file
    with open(f"game_{game_number}.pgn", "w") as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)
if __name__ == "__main__":
    engine_path = '/home/oaltuner/Documents/chess/bora_hoca_kodlar/stockfish/stockfish-ubuntu-x86-64'
    num_games = 50  # Number of games you want to play

    for game_number in range(1, num_games + 1):
        print(f"Starting game {game_number}...")
        play_and_save_game(engine_path, game_number)
        print(f"Game {game_number} completed.")