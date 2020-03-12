from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import imagenet_utils

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16", help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to thier classes inside Keras
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}


# ensure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model commmand line argument should be a key in the 'MODELS' dictionary")

# Typical input images sizes to a CNN trained on ImageNet are:
# 224x224, 227x227, 256x256, 299x299
# initialize input image shape (224x224 pixels) along with pre-processing function
# this might need to be changed based on which model use to classify image
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# if using InceptionV3 or Xception networks, then need to set input shape to (299x299)
# and use different image processing function
if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input


# load the network weights from disk - might take a while when it is done for the first time
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

# load input image using Keras helper utility while ensuring image is resized to 'inputShape'
# the required input dimensions for the ImageNet pre-trained network
print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)

# input image is now represented as a NumPy array of shape (inputShape[0], intputShape[1], 3)
# however need to expand the dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so can pass it through network
image = np.expand_dims(image, axis=0)

# pre-process the image using appropriate function based on the model that has been loaded
image = preprocess(image)

# classify image
print("[INFO] classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)                    # returns the predictions from CNN
P = imagenet_utils.decode_predictions(preds)    # pass to function which return list of ImageNet class label IDs

# loop over the predictions and display the rank-5 predictions + probabilities to terminal
# display top 5 predictions - labels with largest probabilities
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# load image via OpenCV, draw the top prediction on the image, display image to screen
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imwrite('image.png', orig)
cv2.imshow("Classification", orig)
cv2.waitKey(0)










