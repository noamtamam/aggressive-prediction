from sklearn.model_selection import train_test_split
#Import svm model
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import pandas as pd
def run_svm(data, standardize):
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data['win'], test_size=0.3,
                                                        random_state=109)  # 70% training and 30% test
    if standardize:
        trans = StandardScaler()
        X_train = pd.DataFrame(trans.fit_transform(X_train))
        X_test = pd.DataFrame(trans.fit_transform(X_test))
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Import scikit-learn metrics module for accuracy calculation


    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred), "standardize: ", standardize)
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(y_test, y_pred), "standardize: ", standardize)

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred), "standardize: ", standardize)

