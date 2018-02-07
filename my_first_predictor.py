import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import svm

# Trying to predict wine labels usig dataset with 13 features, excluding label.
# Dataset has been imported from sklearn package.
wine = datasets.load_wine();

data = wine.data
target = wine.target
# print (wine.DESCR)

#NOTE: PCA actually reduces accuracy in this case
pca = PCA(n_components = 3)

reduced_data = scale(pca.fit_transform(data))


X_train, X_test, y_train, y_test = train_test_split(reduced_data, wine.target, test_size = 0.25, random_state = 42)

classifier = svm.SVC(kernel='linear')

#--------------------------------------------------------------
#Required only in case of GridSearchCV.
# parameter = [
#             {'C' : [1, 10, 100, 100], 'kernel' : ['linear']},
#             {'C' : [1, 10, 100, 1000], 'gamma' : [1, 0.1, 0.001, 0.0001], 'kernel' : ['rbf']}
#             ]
#---------------------------------------------------------------

if __name__ == '__main__':
    #-----------------------------------------------------------
    # Grid Search Implementation:
    #
    #
    # classifier = GridSearchCV(estimator = svm.SVC(), param_grid=parameter, n_jobs = -1)
    # print ("Best Score" , classifier.best_score_)
    #------------------------------------------------------------

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    score = 0
    for test, pred in zip(y_test, y_pred):
        if (test == pred):
            score += 1
    print ((score / len(y_pred)) * 100)

    # SVM works much better with linear kernel than radial basis function kernel.
    # No kernel tricks required here
