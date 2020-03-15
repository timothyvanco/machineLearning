from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

# initialize the class labels
classLabels = ["cat", "dog", "panda"]

# grab the list of images in the dataset then randomly sample indexes into the image path list
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# initialize the image preprocessors
simple_prep = SimplePreprocessor(32, 32)
Im_to_array = ImageToArrayPreprocessor()

# load dataset from disk then scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[simple_prep, Im_to_array])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# make predictions on the images
print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)
# .predict method of model will return a list of probabilities for every
# image in data – one probability for each class label, respectively
# Taking the argmax on axis=1 finds the index of the class label
# with the largest probability for each image

# loop over sample images
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, display it on screen
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)




