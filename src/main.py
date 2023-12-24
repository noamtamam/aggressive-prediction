import pandas as pd

from make_features import *
from src.visualize import *
from config import *
from ML_models import *


from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    mice_dfs = load_data()
    # plot brain activity for each mice couple
    # for df in mice_dfs:
    #     if df is not None:
    #         plot_heatmap(df, area_names)
    all_data = pd.concat(mice_dfs).reset_index()

    features_data_b_c = all_data[area_names]
    target_data = all_data[target]
    trans = StandardScaler()
    features_data  = pd.DataFrame(trans.fit_transform(features_data_b_c),columns=area_names)


    # print(features_data.describe())
    run_svm(features_data, target_data)
    pca_2d_df, variance_2d = run_PCA(features_data, target_data, 2)
    pca_3d_df, variance_3d = run_PCA(features_data, target_data, 3)
    plot_PCA(pca_2d_df, variance_2d, 2)
    plot_PCA(pca_3d_df, variance_3d, 3)
    pca_2d_df.columns= ['principal component 1', 'principal component 2', target]
    pca_3d_df.columns= ['principal component 1', 'principal component 2',  'principal component 3', target]
    features_data_pca_3 = pca_3d_df[['principal component 1', 'principal component 2',  'principal component 3']]
    target_data_pca_3 = pca_3d_df[target]
    print("svm pca 3 ")
    run_svm(features_data_pca_3, target_data_pca_3)
    features_data_pca_2 = pca_2d_df[['principal component 1', 'principal component 2']]
    target_data_pca_2 = pca_2d_df[target]
    print("svm pca 2")
    run_svm(features_data_pca_2, target_data_pca_2)
    # run PCA
    # visualize PCA
    # run SVM after PCA
    # read on analysis methods on time based data
    # try different SVM models
    # read and do more preprocessing
