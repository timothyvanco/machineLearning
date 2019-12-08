## try to predict the Iris flower species using only two attributes: sepal width and sepal length
## assign a label (a value taken from a discrete set) to an item according to its features

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

#From dataset take first 2 attributes
X, y = X_iris[:, :2], y_iris

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# scaling - For each feature, calculate the average, subtract the mean value from the feature value, and divide the result by their standard deviation
# after scaling - each feature will have a zero average, with a standard deviation of one
# avoid that features with large values may weight too much on the final results
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

colors = ['red', 'greenyellow', 'blue']

for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])

plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width)')