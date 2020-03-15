from lenet_folder import LeNet
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# take MNIST dataset (represented by 284-d vector 28x28 grayscale image)
print("[INFO] accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# channel_first or channel_last
if K.image_data_format() == "channel_first":
    trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
    testData = testData.reshape((testData.shape[0], 1, 28, 28))
else:
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))


# data matrix is properly shaped -> scale data to range [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# convert labels from integers to vectors
LabBin = LabelBinarizer()
trainLabels = LabBin.fit_transform(trainLabels)
testLabels = LabBin.fit_transform(testLabels)

print("[INFO] compiling model...")
optimizerSGD = SGD(lr=0.01)
optimizerAdagrad = Adagrad(lr=0.01)
optimizerAdam = Adam(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimizerAdam, metrics=["accuracy"])

# train network
H = model.fit(trainData, trainLabels,
              validation_data=(testData, testLabels),
              batch_size=128, epochs=1, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testData, batch_size=128)   # variable is NumPy array with shape (len(testX), 10)
print(classification_report(testLabels.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in LabBin.classes_]))

# plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")       # number should be same as number of EPOCHS
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()





