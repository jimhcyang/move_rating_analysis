from collections import deque
import zstandard as zstd
import io
import os

#Takes 80 minutes to execute on January 2023 games (100M).
#The downloaded 33GB of games should be in the assets folder.

def rating_to_group(rating):
    rating = int(rating)
    if rating < 800:
        return 0
    elif rating >= 2400:
        return 9
    return int(rating)//200 - 3
    
def extract_lines(zst_file_path, total_games):
    #total_games = 103,178,407
    annotated_games_count = 0
    total_lines = total_games * 20
    tick = total_lines//20 + 1
    search_string = '[%eval'
    window_size = 20

    lines_window = deque(maxlen=window_size)
    output_chunk = []
    found = False
    not_found = False
    lines_read = 0
    trace_delay = 0

    with open(zst_file_path, 'rb') as zst_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(zst_file) as reader:
            for line in io.TextIOWrapper(reader, encoding='utf-8'):
                    lines_read += 1
                    lines_window.append(line)
                    lines_count = len(lines_window)
                    trace_delay = trace_delay - 1
                    
                    if lines_read % tick == 0:
                        print('{}% completed'.format(5*lines_read//tick))
        
                    if len(line) == 1:
                        trace_delay = 1
        
                    if not trace_delay:
                        if search_string in line:
                            found = True
                        else:
                            not_found = True
                        continue
        
                    if found:
                        candidates = [lines_count-i for i in range(3,lines_count-2)]
                        for candidate in candidates:
                            header = lines_window[candidate]
                            if header.split('"')[0] == "[TimeControl ":
                                tc = header.split('"')[1]
                                break
                        if tc in time_controls:
                            for candidate in candidates:
                                header = lines_window[candidate]
                                if header.split('"')[0] == "[WhiteElo ":
                                    rating = int(header.split('"')[1])
                                    break
                            group = rating_to_group(rating)
                            output_chunk.extend(lines_window)
                            output_file_path = '{}/rating_split_7_rand/group_{}.pgn'.format(asset_folder, group)
                            annotated_games_count += 1
                            with open(output_file_path, 'a', encoding='utf-8') as output_file:
                                output_file.writelines(output_chunk)
        
                    if found or not_found:
                        not_found = False
                        found = False
                        output_chunk = []
                        lines_window = deque(maxlen=window_size)
                        trace_delay = 0

    print("Extraction completed.")
    print("Extracted Percentage is {}".format(annotated_games_count/total_games))

asset_folder = 'asset'
time_controls = {'180+0', '180+2', '300+0', '300+3','600+0', '600+5', '900+10'}
zst_file_path = '{}/lichess_db_standard_rated_2013-01.pgn.zst'.format(asset_folder)


extract_lines(zst_file_path, 103178407)
