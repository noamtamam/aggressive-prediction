from make_features import *
from src.visualize import *
from config import *
from SVM_model import *
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    mice_dfs = load_data()
    for df in mice_dfs:
        if df is not None:
            plot_heatmap(df, area_names)
    all_data = pd.concat(mice_dfs)
    run_svm(all_data)
    # print(all_data.shape)

