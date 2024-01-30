import os
import chess
import chess.pgn
from tqdm import tqdm
from collections import Counter

def sample_games(file_path, file_name, num_games=10000, encoding="utf-8"):
    total_games_seen = 0
    with open(file_path, "r", encoding=encoding) as pgn_file:
        for _ in tqdm(range(num_games), desc="Processing games", unit="games"):
            game = chess.pgn.read_game(pgn_file)
            total_games_seen += 1
            write_to_pgn_file(game, file_name)
    return total_games_seen
    
def write_to_pgn_file(game, output_file, encoding='utf-8'):
    output_path = os.path.join(assets_path, output_file)
    with open(output_path, 'a', encoding='utf-8') as output_pgn:
        output_pgn.write(str(game) + "\n\n")

def count_games(file_path, num_games=1000, start_index=0, encoding="utf-8"):
    total_games_seen = 0
    with open(file_path, "r", encoding=encoding) as pgn_file:
        for _ in tqdm(range(num_games), desc="Processing games", unit="games"):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            total_games_seen += 1
    return total_games_seen

assets_path = os.path.join(os.getcwd(), 'asset')
#NUM_GAMES = 1000
NUM_GAMES = None
while NUM_GAMES is None:
    try:
        n_games = input('Please Enter Number of Games: ')
        if len(n_games) == 0:
            quit()
        NUM_GAMES = int(n_games)
    except ValueError:
        print('Invalid Number! Please Enter Integer.')
        print('Press Enter to Quit.')
        NUM_GAMES = None

file_name = '2023_{}_games.pgn'.format(NUM_GAMES)

for i in range(10):
    sample_size = NUM_GAMES
    out_path = os.path.join(assets_path, 'rating_split_7_rand')
    file_path = os.path.join(out_path, 'group_{}.pgn'.format(i))
    sample_games(file_path, file_name, sample_size)

final_out_path = os.path.join(assets_path, file_name)
print(file_name, count_games(final_out_path, 10*NUM_GAMES))
