from sklearn import datasets

# iris dataset is an object (similar to a dictionary) that has two main components:
# DATA array, TARGET array
iris = datasets.load_iris()

X_iris, y_iris = iris.data, iris.target

print(X_iris.shape, y_iris.shape)
# (150, 4) (150,) - 150 rows, 4 columns

print(X_iris[0], y_iris[0])
# [sepal length, sepal width, petal length, petal width] , [0: setosa, 1: versicolor, and 2: virginica]
# [5.1 3.5 1.4 0.2] 0
