from sklearn.preprocessing import LabelBinarizer
from checkpoint_folder import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import argparse

# construct argument parse and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to best model weights file")
args = vars(ap.parse_args())

# load dataset - training and testing data, then scale it to [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert labels from integers to vectors
libBin = LabelBinarizer()
trainY = libBin.fit_transform(trainY)
testY = libBin.transform(testY)


# initialize optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the callback to sav eonly the best model to the disk based on the validation loss
checkpoint = ModelCheckpoint(args["weights"], monitor="val_loss", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# train network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=2)
