import pandas as pd
import scipy.io
from config import *


"""
parase the timing table csv rows,
givin the details in each row, read the relevante data from the files, 
using load_trial_data()
returns a dictinary with the details and the sliced df the trial of the loser and winner.
"""
def load_data():
    print(MSG_START_LOADING_DATA, end="")
    mice_dfs_lst = []
    mice_info =[]
    timing_table = load_timing_table()
    for idx, row in timing_table.iterrows():
        mouse_1, mouse_2, date, pair, trial_num, winner = load_trial_data(row)
        if mouse_1 is not None:
            mice_info.append({COL_MOUSE_1_DATA: mouse_1,
                              COL_MOUSE_2_DATA: mouse_2,
                              COL_DATE: date,
                              COL_PAIR_NAME: pair,
                              COL_TRIAL_NUM: trial_num,
                              COL_WINNER: winner})
            mice_dfs_lst.extend([mouse_1, mouse_2])
    print(MSG_FINISH_LOADING_DATA)
    return mice_dfs_lst, mice_info

"""
read the csv file and drop irelevant columns 
"""
def load_timing_table() -> pd.DataFrame:
    timing_table = pd.read_csv(timing_table_path)
    timing_table = timing_table.drop(COL_DATA_NOTES, axis=1)
    timing_table[COL_DATA_DATE] = timing_table[COL_DATA_DATE].astype('int').astype('str')
    return timing_table

def load_trial_data(trial_row):
    date, pair, trial_num, start_event, end_event, winner = trial_row[COL_DATA_DATE],\
        trial_row[COL_DATA_PAIR], trial_row[COL_DATA_TRAIL] - 1, trial_row[COL_DATA_START_EVENT],\
        trial_row[COL_DATA_END_EVENT], trial_row['WINNER NUM']
    data_path = mat_file_path.format(date= date,pair= pair)
    try:
        mat = scipy.io.loadmat(data_path)
        # slice the data to the relevante time of trial only
        data = mat[roi_ca_data][:, trial_num, start_event:end_event]
        mouse_1 = pd.DataFrame(data[:n_area, :]).transpose()
        mouse_2 = pd.DataFrame(data[n_area:, :]).transpose()
        mouse_2.columns = area_names
        mouse_1.columns = area_names
        ## sets the win column to indicate in the df if the data is winner or loser
        if winner == 0:
            mouse_1['win'] = 0
            mouse_2['win'] = 1
        else:

            mouse_1['win'] = 1
            mouse_2['win'] = 0
        return mouse_1, mouse_2, date, pair, trial_num, winner
    except Exception as error:
        print(MSG_ERROR_LOADING_DATA, date, pair, trial_num, start_event, end_event)
        print(error)
        return None, None, None, None, None, None

