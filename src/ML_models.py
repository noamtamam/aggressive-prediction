from sklearn.model_selection import train_test_split
#Import svm model
from sklearn import svm
from sklearn import metrics

import pandas as pd
from sklearn.decomposition import PCA
def run_svm(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test

    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Import scikit-learn metrics module for accuracy calculation


    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred) )

def run_PCA(x, y, n_comp):
    pca = PCA(n_components=n_comp)
    principal_components = pca.fit_transform(x)
    total_var = pca.explained_variance_ratio_.sum() * 100
    principal_df = pd.DataFrame(principal_components)
    final_df = pd.concat([principal_df, y], axis=1)
    return final_df, total_var
