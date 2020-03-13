from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from preproces.captcha_helper import preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-m", "--model", required=True, help="path to input model") # path to the serialized weights
args = vars(ap.parse_args())

# load pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# randomly sample a few input images
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

# loop over the image paths
for imagePath in imagePaths:
    # load image and convert it to grayscale, then pad image to ensure digits caught near
    # border of image are retained
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # threshold image to reveal the digits
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find contours in the image keeping only the four largest ones, then sort them from left-to-right
    # return list of (x,y) coordinates that specify the outline of each individual digit
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]  # sorting - keeping only 4 largest outlines
    cnts = contours.sort_contours(cnts)[0]

    # initialize the output image as "grayscale" image with 3 channels along with output prediction
    output = cv2.merge([gray] * 3) #converts to three channel image - R G B
    predictions = []

    # loop over contours
    for c in cnts:
        # compute bounding box for the contour then extract the digit
        (x, y, w, h) = cv2.boundingRect(c)
        roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

        # preprocess the ROI and then classify it
        roi = preprocess(roi, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0] + 1             # add 1 because it starts at 0
        predictions.append(str(pred))

        # draw the prediction on the output image, draw the box around and draw the predicted number
        cv2.rectangle(output, (x-2, y-2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, str(pred), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    print("[INFO] captcha: {}".format("".join(predictions)))
    cv2.imshow("Output", output)
    cv2.waitKey()















