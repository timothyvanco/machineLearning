# https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
from helpers import pyramid
from skimage.transform import pyramid_gaussian
import argparse
import cv2

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# METHOD 1 - no smooth just scaling
# loop over the image pyramid
for (i, resized) in enumerate(pyramid(image, scale=args["scale"])):
    # show resized image
    cv2.imshow("Layer {}".format(i + 1), resized)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# METHOD 2 - resizing + gaussian smoothing
for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
    # if the image is too small, break from the loop
    if resized.shape[0] < 30 or resized.shape[1] < 30:
        break

    cv2.imshow("Layer {}".format(i+1), resized)
    cv2.waitKey(0)

