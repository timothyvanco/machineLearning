import imutils
import cv2

# image - input image I going to pad and resize
# width - target output width
# height - target output height
def preprocess(image, width, height):
    # grab dimensions of the image, then initialize the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise height is greater, than the width so resize along height
    else:
        image = imutils.resize(image, height=height)


    # determine the padding values for the width and height to obtain target dimension
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad image then apply one more resizing to handle any rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))      # ensure all images are in same size

    return image