# USASGE
# python detect_parkinsons.py --dataset dataset/spiral
# python detect_parkinsons.py --dataset dataset/wave

# HOG is a structural descriptor that will capture
# and quantify changes in local gradient in the input image.
# HOG will naturally be able to quantify how the directions of a both spirals and waves change.
# And furthermore, HOG will be able to capture if these drawings have more of a “shake” to them,
# as we might expect from a Parkinson’s patient.

from sklearn.ensemble import RandomForestClassifier 	# classifier
from sklearn.preprocessing import LabelEncoder			# to encode labels as integers
from sklearn.metrics import confusion_matrix			# to derive accuracy, sensitivity, specificity
from skimage import feature								# histogram of orientated gradients (HOG) from feature
from imutils import build_montages						# for visualization
from imutils import paths								# help to extract file paths to each image in dataset
import numpy as np										# easier calculating statistics
import argparse											# to parse command line arguemnts
import cv2												# to read, process and display images
import os												# Unix and Windows file paths with os module


# function to quantify wave/spiral image with HOG method
def quantify_image(image):
	# compute HOG feature vectur for input image
	features = feature.hog(image, orientations=9,
						   pixels_per_cell=(10, 10), cells_per_block=(2, 2),
						   transform_sqrt=True, block_norm="L1")
	return features


# take path to input directory, initialize list of data (images) and class labels
def load_split(path):
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []

	# loop over the image paths
	for imagePath in imagePaths:
		label = imagePath.split(os.path.sep)[-2] 		# extract class label from the filename

		image = cv2.imread(imagePath)						# load input image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)		# convert to grey scale
		image = cv2.resize(image, (200, 200))				# resize to 200x200 pixels

		# threshold image - drawing appears as white on black background
		image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		features = quantify_image(image)

		data.append(features)
		labels.append(label)

	return (np.array(data), np.array(labels))


