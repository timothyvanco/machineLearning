# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    # compute the sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    # compute the derivative of the sigmoid function ASSUMING
    # that the input `x` has already been passed through the sigmoid # activation function
    return x * (1 - x)

def predict(X, W):
    # take the dot product between our features and weight matrix
    pred = sigmoid_activation(X.dot(W))

    # apply a step function to treshold the output to binary class labels
    pred[pred <= 0.5] = 0
    pred[pred > 0] = 1

    # return predictions
    return pred


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type = float, default = 100, help = "# of epochs")
ap.add_argument("-a", "--alpha", type = float, default = 0.01, help = "learning rate")
args = vars(ap.parse_args())

# EPOCHS - number of epochs use when training classifier using gradient descent
# ALPHA - learning rate for gradient descent - typically 0.1, 0.01, 0.001 as initial values

# generate a 2-class classification problem with 1 000 datapoints where each datapoint is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# BIAS TRICK
# insert a column of 1's as the last entry in feature matrix - allow to treat bias as a trainable param with wieght matrix
X = np.c_[X, np.ones((X.shape[0]))]

# split data into test and train - 50% and 50%
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize weight matrix and list of losses - later for plotting
print("[INFO] training...")
W = np.random.rand(X.shape[1], 1)
losses = []

# loop over the desired amount of epochs
for epoch in np.arange(0, args["epochs"]):
    # take dot product between features and weight matrix, then pass value through sigmoid activation function
    pred = sigmoid_activation(trainX.dot(W))
    error = pred - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # gradient descent update is dot product between features and error of the sigmoid derivative of prediction
    d = error * sigmoid_deriv(pred)
    gradient = trainX.T.dot(d)

    # nudge weight matrix in the negative direction of the gradient - taking small step towards set of more optimal params
    W += -args["alpha"] * gradient

    # check to see if update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch{}, loss={:.7f}".format(int(epoch + 1), loss))

# evaluate model
print("[INFO] evaluating...")
pred = predict(testX, W)
print(classification_report(testY, pred))

# plot testing classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()