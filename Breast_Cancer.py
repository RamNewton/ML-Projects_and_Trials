# Objective of program was to familiarise with plotting hyperplane

from sklearn import datasets
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts
from sklearn import svm
import matplotlib.pyplot as plt

# Two functions defined to facilitate the plotting of hyperplane


def make_meshgrid(X, Y, h = 10): # Creates a grid of points that can be passed into contourf function
    x_min = X.min() - 10
    x_max = X.max() + 10
    y_min = Y.min() - 10
    y_max = Y.max() + 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, xx, yy, clf, **params): # Plots the hyperplane using classifier
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy , Z, **params)
    return out

# Sample dataset. breast cancer
cell = datasets.load_breast_cancer()


data = cell.data
y = cell.target

# Extracting out the two underlying principal features so that the hyperplane can be visualised
pca = PCA(n_components=2)
X = pca.fit_transform(data)

X0, X1 = X[:, 0], X[:, 1]

#creating a meshgrid using the two components
xx, yy = make_meshgrid(X0, X1)

# Splitting data, creating linear model
X_train, X_test, y_train, y_test = tts(X, cell.target, test_size = 0.20, random_state = 98)
clf = svm.SVC(kernel = 'linear')
model = clf.fit(X_train, y_train)


fig, sub= plt.subplots(1,1)
ax = sub

#Plotting hyperplane
plot_contours(ax, xx, yy, model, cmap = plt.cm.coolwarm, alpha = 0.8)

# Ploting test data on the graph
ax.scatter(X0, X1, c=y, cmap = plt.cm.coolwarm, s = 10, edgecolors = 'k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_xlabel("Principal Feature 1")
ax.set_ylabel("Principal Feature 2")

# Finally, displaying graph
plt.show()

# Accuracy analysis
predict = clf.predict(X_test)
score = 0
for a, b in zip(predict,y_test):
    if a == b:
        score += 1

print (score*100/len(y_test))
