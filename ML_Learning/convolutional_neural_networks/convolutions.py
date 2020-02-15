from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

def convolve(image, kernel):
    # grab the spatial dimension of the image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for output image, taking care to "pad" borders of input image
    # to the spatial size (width, height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    # loop over the input image, "sliding" the kernel across
    # each (x, y) - coordinate from left-to-right and top-to-bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI (region of interest) of the image by extracting the *center*
            # region of the current (x, y) - coordinates dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform convolution by taking element - wise multiplication between
            # ROI and kernel, then summing the matrix
            k = (roi * kernel).sum()

            # store convolved value in the output (x, y) - coordinate of the output image
            output[y - pad, x - pad] = k

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="int")

# kernel bank, list of kernels to apply
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("emboss", emboss))

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


for (kernelName, K) in kernelBank:
    print("[INFO] applying {} kernel".format(kernelName))
    # apply kernel to gray image using my custom 'convolve' function and 'openCV' function
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)
    
    conv = np.concatenate((gray, convolveOutput, opencvOutput), axis=1)
    cv2.imshow("{} kernel".format(kernelName), conv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
