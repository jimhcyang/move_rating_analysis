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
NUM_GAMES = [15000,10000,10000,15000]
#NUM_GAMES = None
'''
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
'''
file_name = '2023_tc_{}_games.pgn'.format(sum(NUM_GAMES))

for i in range(10):
    sample_size = NUM_GAMES
    #out_path = os.path.join(assets_path, 'rating_split_7_rand')
    out_path = os.path.join(assets_path, 'split_by_tc')
    #file_path = os.path.join(out_path, 'group_{}.pgn'.format(i))
    file_path_180_0 = os.path.join(out_path, 'group_{}_180_0.pgn'.format(i))
    file_path_180_2 = os.path.join(out_path, 'group_{}_180_2.pgn'.format(i))
    file_path_300_0 = os.path.join(out_path, 'group_{}_300_0.pgn'.format(i))
    file_path_600_0 = os.path.join(out_path, 'group_{}_600_0.pgn'.format(i))
    sample_games(file_path_180_0, file_name, sample_size[0])
    sample_games(file_path_180_2, file_name, sample_size[1])
    sample_games(file_path_300_0, file_name, sample_size[2])
    sample_games(file_path_600_0, file_name, sample_size[3])

final_out_path = os.path.join(assets_path, file_name)
#print(file_name, count_games(final_out_path, 10*NUM_GAMES))
print(file_name, count_games(final_out_path, 10*sum(NUM_GAMES)))