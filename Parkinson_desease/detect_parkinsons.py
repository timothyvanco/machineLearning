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

		# extract features
		features = quantify_image(image)

		data.append(features)
		labels.append(label)

	return (np.array(data), np.array(labels)) 	# convert to numpy array and return


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-t", "--trials", type=int, default=5, help="# of trials to run") # default 5 trials
args = vars(ap.parse_args())

# define path to the training and testing directories
trainingPath = os.path.sep.join([args["dataset"], "training"])
testingPath = os.path.sep.join([args["dataset"], "testing"])

# loading training and testing data
print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)

# encode labels as integers
LabEnc = LabelEncoder()
trainY = LabEnc.fit_transform(trainY)
testY = LabEnc.transform(testY)

# initialize trials dictionary
trials = {}

# loop over the number of trials to run
for trial in range(0, args["trials"]):
	print("[INFO] training model {} of {}...".format(trial+1, args["trials"]))
	model = RandomForestClassifier(n_estimators=200)
	model.fit(trainX, trainY)

	# make predictions on testing data and initialize dictionary to store computed metrics
	predictions = model.predict(testX)
	metrics = {}

	# compute confusion matrix and use it to derive the raw accuracy, sensitivity, specificity
	ConMat = confusion_matrix(testY, predictions).flatten()
	(tn, fp, fn, tp) = ConMat
	metrics["accuracy"] = (tp + tn) / float(ConMat.sum())
	metrics["sensitivity"] = tp / float(tp + fn)
	metrics["specificity"] = tn / float(tn + fp)

	# loop over metrics
	for (k, v) in metrics.items():
		# update the trials dictionary with list of values for current metric
		l = trials.get(k, [])
		l.append(v)
		trials[k] = l

# loop over metrics
for metric in ("accuracy", "sensitivity", "specificity"):
	# grab list of values for current metric, compute mean and standard deviation
	values = trials[metric]
	mean = np.mean(values)
	std = np.std(values)

	# show computed metrics for statistic
	print(metric)
	print("=" * len(metric))
	print("u={:.4f}, o={:.4f}".format(mean, std))
	print("")

# create montage - share work visually
# randomly select few images, initialize output for montage
testingPaths = list(paths.list_images(testingPath))
idxs = np.arange(0, len(testingPath))
idxs = np.random.choice(idxs, size=(25,), replace=True)
images = []

# loop over testing samples
for im in idxs:
	# load testing image clone it, resize it
	image = cv2.imread(testingPaths[im])
	output = image.copy()
	output = cv2.resize(output, (128, 128))

	# pre-process image in same manner as earlier
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (200, 200))
	image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	# quantify image and make predictions based on extracted features using trained Random Forest
	features = quantify_image(image)
	preds = model.predict([features])
	label = LabEnc.inverse_transform(preds)[0]

	# draw colored class label on output image and add it to set of output images
	color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
	cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	images.append(output)

# create montage using 128x128 "tiles" with 5 rows and 5 columns
montage = build_montages(images, (128, 128), (5, 5))[0]

# show output montage
cv2.imshow("Output", montage)
cv2.waitKey(0)













