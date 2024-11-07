from make_features import *
from src.visualize import *
from ML_models import *

if __name__ == '__main__':
    mice_dfs, mice_info = load_data()
    #####################################################
    #                   Data exploration                #
    #####################################################
    all_data = pd.concat(mice_dfs).reset_index(drop=True)
    # plot brain activity for each mice couple
    plot_heatmap(mice_info)
    # plot brain activity losser vs winner
    plot_diffreneces(all_data)
    box_plot_area_activity(all_data)
    # PCA
    print("Running and ploting PCA ...", end="")
    # Normalize data
    features_data_b_c = all_data.drop(target, axis=1)
    target_data = all_data[target]
    #
    trans = StandardScaler()
    features_data = pd.DataFrame(trans.fit_transform(features_data_b_c), columns=area_names)
    pca_2d_df, variance_2d, components_2d = run_PCA(features_data, target_data, 2)
    pca_3d_df, variance_3d, components_3d = run_PCA(features_data, target_data, 3)
    plot_PCA(pca_2d_df, variance_2d,components_2d, 2)
    plot_PCA(pca_3d_df, variance_3d,components_3d, 3)
    # print("finished!")
    #####################################################
    #                   Find best model                 #
    #####################################################
    print(features_data.describe())
    find_best_model(mice_dfs)
    # run svm on PCA
    pca_2d_df.columns= ['principal component 1', 'principal component 2', target]
    pca_3d_df.columns= ['principal component 1', 'principal component 2',  'principal component 3', target]
    features_data_pca_3 = pca_3d_df[['principal component 1', 'principal component 2',  'principal component 3']]
    target_data_pca_3 = pca_3d_df[target]
    print("svm pca 3 ")
    run_svm(features_data_pca_3, target_data_pca_3, c=1)
    features_data_pca_2 = pca_2d_df[['principal component 1', 'principal component 2']]
    target_data_pca_2 = pca_2d_df[target]
    print("svm pca 2")
    run_svm(features_data_pca_2, target_data_pca_2, c=1)
    #####################################################
    #                  Run best model                   #
    #####################################################
    svm, m_accuracy = run_svm(features_data, target_data, c=1)
    #####################################################
    #             Explor features importance            #
    #####################################################
    compute_model_significance(svm, features_data, target_data, m_accuracy)
    plot_features_importance(svm)

