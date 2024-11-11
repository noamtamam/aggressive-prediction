#####################################################
#               General Configuration               #
#####################################################

ROI_CA_DATA = 'ROICaData_stim'
NUM_AREA = 24
target = 'win'


area_names = ["DI", "Cl", "Cpu", "Cpu", "AcShv", "AcbC", "AcbSh", "M", "IL", "PrL",
              "PrL", "Cg", "BLA", "CeL", "CPuGP", "Cpu", "SBC", "SBC", "CA",
              "ThlVPM", "ThlVL", "ThlPo", "CA", "DG"]

#####################################################
#               Columns Configuration               #
#####################################################
COL_MOUSE_1_DATA = "mouse_1_data"
COL_MOUSE_2_DATA = "mouse_2_data"
COL_DATE = "date"
COL_PAIR_NAME = "pair_name"
COL_WINNER = "winner"
COL_TRIAL_NUM = "trial_num"

#######################################################
#          Data Columns Configuration                 #
#  (based on the given data sheet - can't be changed) #
#######################################################
COL_DATA_DATE = 'DATE'
COL_DATA_START_EVENT = 'START NEURAL EVENT'
COL_DATA_END_EVENT = 'END NEURAL EVENT'
COL_DATA_TRAIL = 'TRAIL'
COL_DATA_PAIR = 'PAIR'
COL_DATA_NOTES = 'NOTES'

#####################################################
#                  Paths Configuration              #
#####################################################
# Base path for data
BASE_PATH = "../"
DATA_PATH = f"{BASE_PATH}data/"
# File paths
TIMING_TABLE_PATH = f"{DATA_PATH}timing_table.csv"
GRAPH_OUTPUT_PATH = f"{BASE_PATH}graphs/"
MAT_FILE_PATH_TEMPLATE = f'{DATA_PATH}{{date}}/{{pair}}/Matt_files/behaviorVectors_fa_miss.mat'

#####################################################
#             Parameters Configuration              #
#####################################################
N_PERMUTATIONS = 100
ALPHA = 0.05
NUM_FOLDS = 10

#####################################################
#               Messages Configuration              #
#####################################################
MSG_ERROR_LOADING_DATA = "there was an error loading the data for "
MSG_FINISH_LOADING_DATA = "finished loading!"
MSG_START_LOADING_DATA = "Loading the data to dataframes..."



