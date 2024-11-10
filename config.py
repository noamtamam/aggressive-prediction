#####################################################
#                   Configuration                   #
#####################################################

timing_table_path = f"../data/timing_table.csv"
mat_file_path ='../data/{date}/{pair}/Matt_files/behaviorVectors_fa_miss.mat'
roi_ca_data = 'ROICaData_stim'
n_area = 24
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
#               Messages Configuration              #
#####################################################
MSG_ERROR_LOADING_DATA = "there was an error loading the data for "

MSG_FINISH_LOADING_DATA = "finished loading!"

MSG_START_LOADING_DATA = "Loading the data to dataframes..."



