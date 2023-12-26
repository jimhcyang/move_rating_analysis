import chess.pgn
from tqdm import tqdm

date = '2017-05'
output_name = 'asset/analyzed_games_{}.pgn'.format(date)
pgn_file_path = 'asset/lichess_db_standard_rated_{}.pgn'.format(date)

def extract_time_control(tag):
    time_control_str = tag.get("TimeControl", "")
    if time_control_str:
        time_parts = time_control_str.split("+")
        if len(time_parts) > 0:
            try:
                time_value = int(time_parts[0])
                return time_value
            except ValueError:
                pass
    return 0

def read_pgn_file(file_path, max_games=1000, output_file=output_name):
    games_with_eval = []
    total_games_seen = 0
    games_with_eval_seen = 0
    with open(file_path) as pgn_file, open(output_file, 'w') as output_pgn:
        for _ in tqdm(range(max_games), desc="Processing games", unit="games"):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            total_games_seen += 1
            if any("eval" in node.comment for node in game.mainline()):
                time_control = extract_time_control(game.headers)
                if time_control > 150:
                    games_with_eval_seen += 1
                    output_pgn.write(str(game) + "\n\n")
                    output_pgn.flush()

    return total_games_seen, games_with_eval_seen, games_with_eval

total_games_seen, games_with_eval_seen, _ = read_pgn_file(pgn_file_path, max_games=10000000)

# Print the statistics
print(f"Total games seen: {total_games_seen}")
print(f"Games with eval seen: {games_with_eval_seen}")
print(f"Analyzed games saved to 'analyzed_games.pgn'")