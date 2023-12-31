import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import *
from sklearn.svm import SVC
#%%
from joblib import Parallel, delayed
import numpy as np
def run_svm(X, Y, c):
    #Create a svm Classifier
    print("Training SVM model...", end="")
    clf = svm.SVC(kernel='linear', C=c) # Linear Kernel
    num_folds = 10
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cross_val_results  = cross_val_score(clf, X, Y, cv=kf)


    print(f'Cross-Validation Results (Accuracy): {cross_val_results}')
    print(f'Mean Accuracy: {cross_val_results.mean()}')
    observed_accuracy = cross_val_results.mean()
    print("Finished!")
    return clf, observed_accuracy
def get_features_importance(svm_m):
    # Access the coefficients (weights) for each feature
    coefficients = svm_m.coef_[0].tolist()
    sorted_pairs = sorted(zip(coefficients, area_names), key=lambda x: abs(x[0]), reverse=True)
    print("coefficient in descending magnificent order:")
    for pair in sorted_pairs:
        print(pair)

def compute_model_significance(svm_m, X, Y, observed_accuracy):
    n_permutations = 1000
    # Create an array to store permuted accuracies
    permuted_accuracies = np.zeros(n_permutations)
    num_folds = 10
    kf= KFold(n_splits=num_folds, shuffle=True, random_state=42)
    def compute_permuted_accuracy():
        # Shuffle the labels
        shuffled_labels = sklearn.utils.shuffle(Y)

        # Compute accuracy with shuffled labels
        shuffled_scores = cross_val_score(svm_m, X, shuffled_labels, cv=kf)
        return np.mean(shuffled_scores)

    # Parallelize the permutation test
    permuted_accuracies = Parallel(n_jobs=-1)(delayed(compute_permuted_accuracy)() for _ in range(n_permutations))

    # Compute the p-value
    p_value = (np.sum(permuted_accuracies >= observed_accuracy) + 1) / (n_permutations + 1)

    # Print results
    print("Observed Accuracy:", observed_accuracy)
    print("Permuted Accuracies:", permuted_accuracies)
    print("P-value:", p_value)

    # Interpret the results   
    alpha = 0.05  # significance level
    if p_value < alpha:
        print("\nThe observed accuracy is significantly different from random chance.")
    else:
        print("\nThe observed accuracy is not significantly different from random chance.")

def find_best_model(df_data_lst):
    print("Looking for the best model..")
    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear']
    }

    # Initialize the SVM model
    svm_model = SVC()

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')

    # Iterate over the combinations of parameters
    for chunk_size in [1, 2, 5, 10, 20, 30, 40, 50]:
        # Resample data by chunk size
        resampled_dfs = resample_dfs(chunk_size, df_data_lst)
        all_data = pd.concat(resampled_dfs).reset_index(drop=True)

        X_not_scaled = all_data[area_names]
        y = all_data[target]
        # Scale the data
        trans = StandardScaler()
        X = pd.DataFrame(trans.fit_transform(X_not_scaled), columns=area_names)

        # Split your data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the model with the current parameters
        grid_search.fit(X_train, y_train)

        # Print the results for the current chunk size
        print(
            f"Chunk Size: {chunk_size}, Best Parameters: {grid_search.best_params_}, Best Score: {grid_search.best_score_}")

    # Get the best parameters from the grid search
    best_params = grid_search.best_params_

    # Train the final model using the best parameters on the entire training set
    final_model = SVC(**best_params)
    final_model.fit(X_train, y_train)

    # Evaluate the final model on the test set
    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Finished! Final Model Accuracy: {accuracy}")

def run_PCA(x, y, n_comp):
    pca = PCA(n_components=n_comp)
    principal_components = pca.fit_transform(x)
    total_var = pca.explained_variance_ratio_.sum() * 100
    principal_df = pd.DataFrame(principal_components)
    final_df = pd.concat([principal_df, y], axis=1)
    return final_df, total_var

def resample_dfs(resample_size: int, dfs_lst: list) -> list:
    resampled_dfs = []
    for df in dfs_lst:
        if df is not None:
            df_sub = df.rolling(resample_size).mean()[::resample_size]
            df_sub=df_sub.dropna().reset_index(drop=True)
            resampled_dfs.append(df_sub)
    return resampled_dfs
