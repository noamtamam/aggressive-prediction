import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.io
from config import *
import mat73


def load_data(standardize=True):
    mice_dfs = []
    timing_table = load_timing_table()
    for idx, row in timing_table.iterrows():
        mouse_1, mouse_2 = load_trial_data(row, standardize)
        mice_dfs.extend([mouse_1, mouse_2])
    return mice_dfs

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
        mouse_1_data = np.transpose(data[n_area:,:])
        mouse_2_data = np.transpose(data[:n_area,:])
        winner_labels = np.ones((24, 1, 1), dtype=int)
        losser_labels = np.zeros((24, 1, 1), dtype=int)

        # mouse_1 = pd.DataFrame(data[:n_area,:]).transpose()
        # mouse_2 = pd.DataFrame(data[n_area:,:]).transpose()
        # mouse_2.columns = area_names
        # mouse_1.columns = area_names

        if winner == 0:
            mouse_1 = np.concatenate((mouse_1_data[:, :, np.newaxis], winner_labels), axis=2)
            mouse_2 = np.concatenate((mouse_2_data[:, :, np.newaxis], losser_labels), axis=2)
            # mouse_1['win'] = 0
            # mouse_2['win'] = 1
        else:
            mouse_1 = np.concatenate((mouse_1_data[:, :, np.newaxis], losser_labels), axis=2)
            mouse_2 = np.concatenate((mouse_2_data[:, :, np.newaxis], winner_labels), axis=2)
            # mouse_1['win'] = 1
            # mouse_2['win'] = 0
        return mouse_1, mouse_2
    except Exception as error:
        # print(data_path)
        # print(error)
        return None, None