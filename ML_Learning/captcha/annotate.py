from imutils import paths
import argparse
import imutils
import cv2
import os


# --input : input path to raw captcha images - downloads directory
# --annot : output path to where will be storing the labeled digits - dataset directory
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory")
ap.add_argument("-a", "--annot", required=True, help="path to output directory of annotations")
args = vars(ap.parse_args())


# grab the image paths then initialize the directory of character counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}



# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # display an update to the user
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

    # load image and convert it to grayscale, then pad image
    # (with 8 pixels in every direction) to ensure digits caught
    # on the border of the image are retained (in case any digit is touching the border)
    # threshold the image to reveal the digits
    # find contours in the image, keeping only the four largest ones
    try:
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        for c in cnts:  # extract each contour
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h, x - 5:x + w + 5]

            # display the character, making it large enough to see, wait for keypress
            # ROI - region of interest
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)    # the pressed key will be used as the label for the digit
            # if '-' key is pressed - ignore the character
            if key == ord("-"):
                print("[INFO] ignoring character")
                continue

            # grab the key that was pressed and construct the path to output directory
            key = chr(key).upper()
            dirPath = os.path.sep.join([args["annot"], key])

            # if the output directory does not exist, create it
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            # write labeled character to file
            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)

            # increment the count for the current key
            counts[key] = count + 1


    # trying to control-c out of the script, so break from the loop (still need to press a key
    # for the active window to trigger this)
    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break

    # an unknown error has occured for this particular image
    except:
        print("[INFO] skipping image...")



