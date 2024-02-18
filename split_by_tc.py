import chess.pgn
from tqdm import tqdm
import os

def process_pgn_file_with_chess(input_file, output_folder, total_games):
    # Ensure the output folder exists
    root_file_name = input_file.split('.')[0]
    os.makedirs(output_folder, exist_ok=True)
    
    # Define the valid time controls
    valid_time_controls = {'180+0', '180+2', '300+0', '600+0'}

    # Initialize a dictionary to hold games by their time control
    games_by_time_control = {tc: [] for tc in valid_time_controls}

    with open(input_file, 'r') as pgn:
        # Since we can't know the total number of games upfront, use tqdm with manual update
        pbar = tqdm(desc="Processing games")
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break  # Exit the loop if there are no more games in the file

            pbar.update(1)  # Update progress bar per game processed

            time_control = game.headers['TimeControl']
            if time_control in valid_time_controls:
                exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
                pgn_string = game.accept(exporter)
                games_by_time_control[time_control].append(pgn_string)
        pbar.close()

    for time_control, games_list in games_by_time_control.items():
        if games_list:
            output_file_name = os.path.join(output_folder, f'{root_file_name}_{time_control.replace("+", "_")}.pgn')
            with open(output_file_name, 'w') as output_file:
                output_file.write('\n\n\n'.join(games_list))

assets_path = os.path.join(os.getcwd(), 'asset')
games_path = os.path.join(assets_path, 'rating_split_7_rand')
games_totes = [154705, 484739, 848927, 1097034, 1275576, 1171586, 966077, 617368, 354205, 244783]

inputs = ['group_{}.pgn'.format(i) for i in range(10)]
for i, input_file_path in enumerate(inputs):
    print(f'processing {input_file_path}')
    root_file_path = os.path.join(games_path, input_file_path)
    output_folder_path = os.path.join(assets_path, 'split_by_tc')  # The folder where you want to save the output files
    process_pgn_file_with_chess(root_file_path, output_folder_path, games_totes[i])
