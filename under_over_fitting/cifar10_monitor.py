# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from training_monitor_folder import TrainingMonitor
from training_monitor_folder import MiniVGGNet

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output directory")
args = vars(ap.parse_args())

print("[INFO] process ID: {}".format(os.getpid()))

# load training and testing data, scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert labels from integers to vectors
labBin = LabelBinarizer()
trainY = labBin.fit_transform(trainY)
testY = labBin.transform(testY)

# initialize label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath = jsonPath)]

# train network
print("[INFO] training network...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=100, callbacks=callbacks, verbose=1)

