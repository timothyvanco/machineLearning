from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19.model", help="path to ouput loss/accuracy model")
args = vars(ap.parse_args())


# initialize initial learning rate, number of epochs to train for and batch size
INIT_LR = 0.001
EPOCHS = 25
BS = 8


# grab list of images in our dataset directory, then initialize list of data (images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []


# loop over the image paths
for imagePath in imagePaths:
    # extract the class label (covid/normal) from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load image, swap color channels, resize to be fixed 224x224 pixels, ignore aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # update data and labels list
    data.append(image)
    labels.append(label)

# convert data and labels to NumPy arrays while scaling the pixel intensities to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)


# one-hot encoding on the labels
# - each encoded label consist of 2 element array with one element "hot" = 1 versus "not" = 0
# create a vector - [1, 0] - for first class, [0, 1] - second class (from names -> binary 1/0 in vector)
LabBin = LabelBinarizer()
labels = LabBin.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 80% - training, 20% - testing
# set random_state to specific number - always will be same result in splitting for other tests in future
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# initialize training data augmentation object
# WHY?
trainAugmentation = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

# load VGG16 network, ensuring head FC layers sets are left off
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# construct head of the model that will be placed on top of the base model
# WHY?
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place head FC model on top of the base model (this will become actual model to train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them, so they wont update during first training process
# WHY this?
for layer in baseModel.layers:
    layer.trainable = False
    
    
# compile model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# binary_crossentropy - 2 class problem (health or sick patient)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train head of the network
H = model.fit_generator(
    trainAugmentation.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)


# evaluation
# make prediction on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in testing set need to find index of label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=LabBin.classes_))

# compute confusion matrix and use it to derive raw accuracy, sensitivity, specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show confusion matrix, accuracy, sensitivity, specificity
print(cm)
print("accuracy: {:.4f}".format(accuracy))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# plot training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize model to disk - save it
print("[INFO] saving COVID-19 detector model...")
model.save(args["model"], save_format="h5")












