import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.io
from config import *
import mat73


def load_data(standardize=True):
    print("Loading the data to dataframes...", end="")
    mice_dfs_lst = []
    mice_info =[]
    timing_table = load_timing_table()
    for idx, row in timing_table.iterrows():
        mouse_1, mouse_2, date, pair, trial_num = load_trial_data(row, standardize)
        if mouse_1 is not None:
            mice_info.append({"mouse_1_data": mouse_1,
                              "mouse_2_data": mouse_2,
                              "date": date,
                              "pair_name": pair,
                              "trial_num": trial_num})
            mice_dfs_lst.extend([mouse_1, mouse_2])
    print("finished!")
    return mice_dfs_lst, mice_info

def load_timing_table() -> pd.DataFrame:
    timing_table = pd.read_csv(timing_table_path)
    timing_table = timing_table.drop('NOTES', axis=1)
    timing_table['DATE'] = timing_table['DATE'].astype('int').astype('str')
    return timing_table

def load_trial_data(trial_row, standardize):
    date, pair, trial_num, start_event, end_event, winner = trial_row['DATE'],\
        trial_row['PAIR'], trial_row['TRAIL']-1, trial_row['START NEURAL EVENT'],\
        trial_row['END NEURAL EVENT'], trial_row['WINNER NUM']

    data_path = mat_file_path.format(date= date,pair= pair)
    try:
        mat = scipy.io.loadmat(data_path)
        data = mat[roi_ca_data][:, trial_num, start_event:end_event]

        mouse_1 = pd.DataFrame(data[:n_area,:]).transpose()
        mouse_2 = pd.DataFrame(data[n_area:,:]).transpose()
        mouse_2.columns = area_names
        mouse_1.columns = area_names
        # get mean of each area
        # mouse_2 = pd.DataFrame(mouse_2.mean()).transpose()
        # mouse_1 = pd.DataFrame(mouse_1.mean()).transpose()
        if winner == 0:
            mouse_1['win'] = 0
            mouse_2['win'] = 1
        else:

            mouse_1['win'] = 1
            mouse_2['win'] = 0
        return mouse_1, mouse_2, date, pair, trial_num
    except Exception as error:
        # print(data_path)
        # print(error)
        return None, None, None, None, None

