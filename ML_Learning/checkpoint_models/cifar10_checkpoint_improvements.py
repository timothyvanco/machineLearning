from sklearn.preprocessing import LabelBinarizer
from checkpoint_folder import MiniVGGNet

# enable to checkpoint and serialize network to disk whenever find an improvement in model performance
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import argparse
import os

# will store weights during training process
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to weights directory")
args = vars(ap.parse_args())

# load dataset - training and testing data, then scale it to [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert labels from integers to vectors
labBin = LabelBinarizer()
trainY = labBin.fit_transform(trainY)
testY = labBin.transform(testY)

# initialize optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the callback to save only the *best* model to disk based on validation loss
fname = os.path.sep.join([args["weights"],"weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
# monitor - what I am monitoring
# mode - max or min - because I am monitoring loss - lower is better -> min (val_acc -> max)
# save_best_only - of course true
# verbose - simply logs a notification to terminal whe na model is being serialized to disk
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# train network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=2)










