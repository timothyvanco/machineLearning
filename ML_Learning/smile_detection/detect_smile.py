from tensorflow.keras.preprocessing.image import img_to_array   # convert each frame from video to array
from tensorflow.keras.models import load_model                  # to load weights of pre-trained LeNet
import numpy as np
import argparse
import imutils
import cv2

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# cascade - based on Haar cascade algorithm - capable of detecting objects regardless of their location and scale
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# load face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

# if a video path was not supplied, grab reference to webcam
# otherwise, load a video
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])


# loop until stop a script or end of video file
while True:
    (grabbed, frame) = camera.read()    # grab current frame

    # if viewing video and we did not grab a frame, reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize frame, convert it to grayscale, clone original frame to fraw it later
    frame = imutils.resize(frame, width=300)
    frame = cv2.flip(frame, 0)                 # rotate 180 degrees
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    # detect faces in the input frame, then clone frame to draw on it
    # consider a face - must have a minimum width 30x30 pixels
    # minNeighbosr - helps prune false-positives
    # scaleFactor - controls the number of image pyramid levels generated
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over each set of bounding boxes
    for (fX, fY, fW, fH) in rects:
        roi = gray[fY:fY + fH, fX:fX + fW]      # extract ROI of the face from grayscale image
        roi = cv2.resize(roi, (28, 28))         # resize it to fixed 28x28 pixels
        roi = roi.astype("float") / 255.0       # prepare for classification for CNN
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        (notSmiling, smiling) = model.predict(roi)[0] # determine prob of "smiling"/"notsmiling"
        label = "Smiling" if smiling > notSmiling else "Not Smiling"

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    # show detected faces along with smiling/not smiling labels
    cv2.imshow("Face", frameClone)

    # if the 'q' is pressed, stop loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup camera and close any open window
camera.release()
cv2.destroyAllWindows()













