# %%
import chess
import chess.pgn
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

# %%
FILENAME = None
asset_dir = 'asset'
assets_path = os.path.join(os.getcwd(), asset_dir)

def contains_file(file_name, directory=assets_path):
    # Check if the directory exists
    if not os.path.isdir(directory):
        print('directory does not exist')
        return False
    
    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        if filename == file_name:
            return True
    print('file does not exist')
    return False

while FILENAME is None:
    print('Please provide a valid pgn file with only one game that has %clk and %eval')
    pgn_file_name = input('Please Enter PGN File Name: ')
    if len(pgn_file_name) == 0:
        quit()
    if contains_file(pgn_file_name):
        FILENAME = pgn_file_name
        bwg = None
        while bwg is None:
            print('Please provide the color of the player')
            bwg = input('[b/w/g] for black rating, white rating, or game rating: ')
            if bwg in {'b', 'w', 'g'}:
                break
            else:
                bwg = None
                print('Invalid character!')
        break
    print('Invalid PGN file! File not in Directory "asset".')
    print('Press Enter to Quit.')
    FILENAME = None

skips = 1 if bwg == 'g' else 2
start = 1 if bwg == 'b' else 0


file_name = FILENAME
single_path = os.path.join(assets_path, file_name)

# %%
from chess_class import ChessGame, ChessMove

# %%
def chess_games_to_arrays(games_generator):
    def rating_to_group(rating):
        rating = int(rating)
        if rating < 800:
            return 0
        elif rating >= 2400:
            return 9
        return int(rating)//200 - 3

    attributes = ["ply_count", "time_category", "classification_name", "count_legal_moves", "force_moves_percent",
                  "game_state", "distance", "is_endgame", "has_increment", "in_time_trouble", "can_dirty_flag",
                  "is_check", "is_double_check", "is_discovered_check", "is_capture", "is_threat", "is_developing",
                  "is_retreating", "was_hanging", "is_hanging", "was_true_hanging", "is_true_hanging", "is_create_tension",
                  "is_resolve_tension", "is_maintain_tension", "start_square", "end_square", "threats", 
                  "create_tension", "maintain_tension", "resolve_tension", "piece_value"]
    
    game_arrays = []
    ratings_list = []
    urls_list = []
    for i, game in enumerate(tqdm(games_generator, total=max_games, desc="Processing games")):
        elo_w, elo_b, url = game.white_elo, game.black_elo, game.url
        total_plies = game.total_ply
        df = pd.DataFrame(columns=attributes)
        for j, move in enumerate(game.moves):
            move_row = {attribute: getattr(move, attribute, None) for attribute in attributes}
            df.loc[j] = move_row
        df['ply_count'] = df['ply_count'] / total_plies
        df['count_legal_moves'] = df['count_legal_moves'] / 128
        df['distance'] = (df['distance'] - 1) / 6

        df['prev_end_square'] = df['end_square'].shift(1).fillna(64)
        df['prev_threats'] = df['threats'].shift(1).fillna({}).apply(lambda x: x if isinstance(x, set) else {})
        df['prev_create_tension'] = df['create_tension'].shift(1).fillna({}).apply(lambda x: x if isinstance(x, set) else {})
        df['last_move_end_square'] = df['end_square'].shift(2).fillna(64)
        df['last_move_create_tension'] = df['create_tension'].shift(2).fillna({}).apply(lambda x: x if isinstance(x, set) else {})
        df['last_move_threats'] = df['threats'].shift(2).fillna({}).apply(lambda x: x if isinstance(x, set) else {})
    
        df['is_reacting'] = df.apply(lambda row: row['prev_end_square'] in (row['create_tension'] | row['threats']), axis=1) | \
                            (df['prev_end_square'] == df['end_square']) | \
                            df.apply(lambda row: row['start_square'] in row['prev_threats'], axis=1)
        df['is_same_piece'] = df['last_move_end_square'] == df['start_square']
        df['veni_vidi_vici'] = df.apply(lambda row: row['end_square'] in (row['last_move_create_tension'] | row['last_move_threats']), axis=1)
        df['is_collinear'] = df.apply(lambda row: row['start_square'] in (row['prev_create_tension'] | row['prev_threats']), axis=1) | \
                            df.apply(lambda row: row['prev_end_square'] in row['create_tension'], axis=1)
        df.drop(columns=['prev_end_square', 'last_move_end_square', 'prev_threats', 'last_move_create_tension', 'prev_create_tension',
                         'last_move_threats', 'threats', 'create_tension', 'maintain_tension', 'resolve_tension'], inplace=True)

        df['moved_piece_king'] = df['piece_value'].apply(lambda x: 1 if x == 6 else 0)
        df['moved_piece_queen'] = df['piece_value'].apply(lambda x: 1 if x == 5 else 0)
        df['moved_piece_rook'] = df['piece_value'].apply(lambda x: 1 if x == 4 else 0)
        df['moved_piece_bishop'] = df['piece_value'].apply(lambda x: 1 if x == 3 else 0)
        df['moved_piece_knight'] = df['piece_value'].apply(lambda x: 1 if x == 2 else 0)
        df['moved_piece_pawn'] = df['piece_value'].apply(lambda x: 1 if x == 1 else 0)
        df['time_category_instant'] = df['time_category'].apply(lambda x: 1 if x == 'instant' else 0)
        df['time_category_fast'] = df['time_category'].apply(lambda x: 1 if x == 'fast' else 0)
        df['time_category_normal'] = df['time_category'].apply(lambda x: 1 if x == 'normal' else 0)
        df['time_category_slow'] = df['time_category'].apply(lambda x: 1 if x == 'slow' else 0)
        df['classification_name_Great'] = df['classification_name'].apply(lambda x: 1 if x == 'Great' else 0)
        df['classification_name_Good'] = df['classification_name'].apply(lambda x: 1 if x == 'Good' else 0)
        df['classification_name_Inaccuracy'] = df['classification_name'].apply(lambda x: 1 if x == 'Inaccuracy' else 0)
        df['classification_name_Blunder'] = df['classification_name'].apply(lambda x: 1 if x == 'Blunder' else 0)
        df['classification_name_Mistake'] = df['classification_name'].apply(lambda x: 1 if x == 'Mistake' else 0)

        df = df.drop(['classification_name', 'time_category', 'piece_value', 'start_square', 'end_square'], axis=1)

        game_array_rep = df.astype(float).to_numpy()
        game_arrays.append(game_array_rep)
        ratings_list.append(rating_to_group((elo_w + elo_b)/2))
        urls_list.append(url)
    return game_arrays, ratings_list, urls_list

# %%
def load_pgns(file_path, num_games=None, start_index=0, encoding="utf-8"):
    games = []
    with open(file_path, "r", encoding=encoding) as file:
        for _ in tqdm(range(start_index), desc='Skipping games', unit='game', leave=False):
            game = chess.pgn.read_game(file)
            if game is None:
                break
        for _ in tqdm(range(num_games), desc='Loading games', unit='game', leave=True) if num_games else iter(int, 1):
            game = chess.pgn.read_game(file)
            if game is None:
                break
            games.append(game)
    return games

# %%
max_games = 1
pgns = load_pgns(single_path, max_games)
games_generator = (ChessGame(pgn) for pgn in pgns)
game_arrays, ratings_list, urls_list  = chess_games_to_arrays(games_generator)

# %%
input_size = 42
hidden_size = 128
num_classes = 10
num_epochs = 12
num_layers = 2
learning_rate = 0.001
dropout_rate = 0
sequence_length = len(game_arrays[0])
batch_size = 1
alpha = 0.64

# %%
def pad_game(game, max_length=256, vector_size=42):
    padding_length = max_length - len(game)
    if padding_length < 0:
        return game[:max_length]
    else:
        padding = np.full((padding_length, vector_size), -1)
        return np.vstack((game, padding))

padded_game = [pad_game(g, sequence_length)[start::skips] for g in game_arrays]
padded_games = [padded_game[0][:i+1] for i in range(len(padded_game[0]))]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc_classification = nn.Linear(hidden_size, num_classes)
        self.fc_regression = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 

        out, _ = self.lstm(x, (h0,c0))  
        out = out[:, -1, :]
        
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.dropout(F.relu(self.fc3(out)))
        out = self.dropout(F.relu(self.fc4(out)))
        out = self.dropout(F.relu(self.fc5(out)))
        out = self.dropout(F.relu(self.fc6(out)))
        out = self.dropout(F.relu(self.fc7(out)))
        classification_output = self.fc_classification(out)
        regression_output = self.fc_regression(out)
        return classification_output, regression_output

# %%
model_path = '2023_tc_50000_games_pred.pth'
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device);

# %%
for m, test_data in enumerate(padded_games):
    moves = torch.FloatTensor(np.array([test_data]))
    #print('Move {}'.format(m+1))
    model.eval()
    with torch.no_grad():
        moves = moves.to(device)
        classification_output, _ = model(moves)
        probabilities = F.softmax(classification_output, dim=1)
        probs = probabilities.cpu().numpy()
        pred = torch.max(classification_output.data, 1)

    cusum = 0
    for i, p in enumerate(probs[0]):
        if i == 0:
            if p < .16:
                cusum += 700*p
            elif p < .32:
                cusum += (350*p + 56)
            else:
                cusum += 168
        elif i == 9:
            if p < .16:
                cusum += 2500*p
            elif p < .32:
                cusum += (2850*p - 56)
            else:
                cusum += (3200*p - 168)
        else:
            cusum += (700+200*i)*p
    #print('Predicted Rating: {:.0f}'.format(cusum))

print('Model Prediction: {}'.format(pred[1].item()))
print('True Class: {}'.format(ratings_list[0]))
for i, p in enumerate(probs[0]):
    print('Class {} Probabilities: {:.10f}'.format(i, p))