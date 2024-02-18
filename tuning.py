# %%
import chess
import chess.pgn
import csv
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
from sklearn.preprocessing import StandardScaler
from collections import Counter

from chess_class import ChessGame, ChessMove

# %%
## Which PGN File To Train
max_games = 500000 #100000
asset_dir = 'asset'
file_name = '2023_10000_games.pgn'
file_name = '2023_tc_50000_games.pgn'

# %%
def save_item_to_file(games, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(games, file)

def load_item_from_file(file_path):
    if os.path.exists(file_path):
        print('loading item from cache...')
        with open(file_path, 'rb') as file:
            items = pickle.load(file)
        print('loaded')
        return items
    else:
        return None

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
pgns = None
assets_path = os.path.join(os.getcwd(), asset_dir)
single_path = os.path.join(assets_path, file_name)

cached_pgns_file = file_name.split('.')[0] + '_pgn.pkl'
cached_urls_file = file_name.split('.')[0] + '_urls_list.pkl'
cached_ratings_file = file_name.split('.')[0] + '_ratings_list.pkl'
cached_games_file = file_name.split('.')[0] + '_game_arrays.pkl'
cached_pgns_path = os.path.join(assets_path, cached_pgns_file)
cached_urls_path = os.path.join(assets_path, cached_urls_file)
cached_ratings_path = os.path.join(assets_path, cached_ratings_file)
cached_games_path = os.path.join(assets_path, cached_games_file)

chess_games_loaded = True
urls_list = load_item_from_file(cached_urls_path)
ratings_list = load_item_from_file(cached_ratings_path)
game_arrays = load_item_from_file(cached_games_path)

if ratings_list is None:
    print('Creating new ratings_list and urls_list...')
if game_arrays is None:
    print('Creating new game_arrays...')
    chess_games_loaded = False

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
        #ratings_list.append([elo_w, elo_b])
        urls_list.append(url)
    return game_arrays, ratings_list, urls_list

# %%
if not chess_games_loaded:
    pgns = load_item_from_file(cached_pgns_path)
    if pgns is None:
        pgns = load_pgns(single_path, max_games)
        save_item_to_file(pgns, cached_pgns_path)
    #games_generator = (ChessGame(pgn) for pgn in pgns)
    games_generator = (ChessGame(pgn) for i, pgn in enumerate(pgns) if i % 5 == 0)
    game_arrays, ratings_list, urls_list  = chess_games_to_arrays(games_generator)
    save_item_to_file(game_arrays, cached_games_path)
    save_item_to_file(ratings_list, cached_ratings_path)
    save_item_to_file(urls_list, cached_urls_path)

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
def combined_loss(classification_output, regression_output, target, alpha=0.5):
    classification_loss = nn.CrossEntropyLoss()(classification_output, target)
    regression_target = target.float()
    regression_loss = nn.MSELoss()(regression_output.squeeze(), regression_target)
    return alpha * classification_loss + (1 - alpha) * regression_loss

def train_model(model, train_loader, test_loader, optimizer, num_epochs, device, alpha=0.5):
    torets = []
    for epoch in range(num_epochs):
        model.train()
        for i, (moves, labels) in enumerate(train_loader):  
            moves = moves.to(device)
            labels = labels.to(device)

            classification_output, regression_output = model(moves)
            loss = combined_loss(classification_output, regression_output, labels, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        predicted_probs, predicted_labels, actual_labels = test_model(model, test_loader, device)
        pred_closeness = [sum(abs(p - a) <= k for p, a in zip(predicted_labels, actual_labels)) for k in range(10)]
        toret = [x/20000 for x in pred_closeness]
        torets.append(toret)
    return torets

def test_model(model, test_loader, device):
    model.eval()
    n_correct = 0
    n_samples = 0
    predicted_probs = []
    predicted_labels = []
    actual_labels = []
    with torch.no_grad():
        for moves, labels in test_loader:
            moves = moves.to(device)
            labels = labels.to(device)
            classification_output, _ = model(moves)
            probabilities = F.softmax(classification_output, dim=1)

            _, predicted = torch.max(classification_output.data, 1)
            predicted_probs.extend(probabilities.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the test moves: {acc} %')
    return predicted_probs, predicted_labels, actual_labels

# PIECE: [0, 1, 2, 4, 5, 9, 10, 11, 12, 14, 15, 27, 28, 29, 30, 31, 32]
# TIME: [6, 7, 8, 33, 34, 35, 36]
# ENGINE: [3, 37, 38, 39, 40, 41]
# DOMAIN: [13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

def get_loaders(padded_games, ratings_list, urls_list, batch_size, fold_number=0):
    if fold_number < 0 or fold_number > 4:
        raise ValueError("fold_number must be between 0 and 4")
    test_list = padded_games[fold_number::5]
    #print(len(test_list))
    train_list = [df for i in range(5) if i != fold_number for df in padded_games[i::5]]
    test_ratings = ratings_list[fold_number::5]
    train_ratings = [ratings for i in range(5) if i != fold_number for ratings in ratings_list[i::5]]
    test_urls = urls_list[fold_number::5]
    train_urls = [url for i in range(5) if i != fold_number for url in urls_list[i::5]]

    train_data = [torch.FloatTensor(doc) for doc in train_list]
    test_data = [torch.FloatTensor(doc) for doc in test_list]
    train_labels = torch.LongTensor(train_ratings)
    test_labels = torch.LongTensor(test_ratings)

    train_dataset = TensorDataset(torch.stack(train_data), train_labels)
    test_dataset = TensorDataset(torch.stack(test_data), test_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_urls, test_urls


def pad_game(game, max_length=256, vector_size=42):
    padding_length = max_length - len(game)
    if padding_length < 0:
        return game[:max_length]
    else:
        padding = np.full((padding_length, vector_size), -1)
        return np.vstack((game, padding))

# Hyperparameter ranges
'''
input_size = 42
num_classes = 10
num_epochs = 42
sequence_lengths = [80] #1
batch_sizes = [100] #3
hidden_sizes = [128] #1
num_layers_list = [2] #1
learning_rates = [0.001] #1
alphas = [0.8] #4
dropout_rates = np.arange(0,0.51,0.05) #3
decays = [0.000001, 0.0000016, 0.000004, 0.0000064, 0.00001, 0.000016, 0.00004, 0.000064, 0.0001] #3
torch.manual_seed(64)
'''
input_size = 42
num_classes = 10
num_epochs = 42
sequence_lengths = [100] #8
batch_sizes = [100] #7
hidden_sizes = [100] #4
num_layers_list = [2,3] #5
learning_rates = [3.2e-4, 1e-3, 3.2e-3] #6
alphas = [0.575, 0.875] #3
dropout_rates = [0.05,0.25] #2
decays = [1e-4] #1
torch.manual_seed(64)

csv_file = "hyperparameter_tuning_results.csv"
csv_columns = ['decay', 'dropout_rate', 'alpha_value', 'hidden_size', 'num_layers', 'learning_rate', 'batch_size', 'sequence_length', 'fold_number', 'lists']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
        
total_configs = len(sequence_lengths) * len(batch_sizes) * 5 * len(hidden_sizes) * len(num_layers_list) * len(learning_rates) * len(dropout_rates) * len(alphas) * len(decays)
current_config = 0

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for decay in decays:
        for dropout_rate in dropout_rates:
            for alpha_value in alphas:
                for hidden_size in hidden_sizes:
                    for num_layers in num_layers_list:
                        for learning_rate in learning_rates:
                            for batch_size in batch_sizes:
                                for sequence_length in sequence_lengths:
                                    padded_games = [pad_game(g, sequence_length) for g in game_arrays]
                                    for fold_number in range(5):
                                        train_loader, test_loader, _, _ = get_loaders(padded_games, ratings_list, urls_list, batch_size, fold_number)

                                        # Increment the configuration counter
                                        current_config += 1
                                        print(f'Testing configuration {current_config} out of {total_configs}: Sequence Length={sequence_length}, Batch Size={batch_size}, Fold Number={fold_number}, Hidden Size={hidden_size}, Num Layers={num_layers}, Learning Rate={learning_rate}, Dropout Rate={dropout_rate}, Alpha={alpha_value}, Decay={decay}')
                                        
                                        model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
                                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
                                        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                                        print(f'The model has {num_params:,} trainable parameters')

                                        lists = train_model(model, train_loader, test_loader, optimizer, num_epochs, device, alpha_value)

                                        row = {
                                            'decay': decay,
                                            'dropout_rate': dropout_rate,
                                            'alpha_value': alpha_value,
                                            'hidden_size': hidden_size,
                                            'num_layers': num_layers,
                                            'learning_rate': learning_rate,
                                            'batch_size': batch_size,
                                            'sequence_length': sequence_length,
                                            'fold_number': fold_number,
                                            'lists': lists
                                        }
                                        writer.writerow(row)