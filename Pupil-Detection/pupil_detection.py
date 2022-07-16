"""
Created on Wed Sep 29 09:50:06 2021

@author: Brendan Bassett
@Date: 10/6/2021

Written for CS3150 Image Processing
Dr. Feng Jiang
Metropolitan State University of Denver
"""

import math
import os

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plot
from scipy.signal import convolve2d

# ======================================================================================================================
# Setup
# ======================================================================================================================

TEST_IMG = "iris"
JPG = ".jpg"
BMP = ".bmp"

ROOT_DIRECTORY = os.path.abspath('.')
print("ROOT DIRECTORY", ROOT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.abspath(os.path.join("output"))
print("OUTPUT DIRECTORY", OUTPUT_DIRECTORY)

NONE = 0
OPENCV = 1
PYPLOT = 2


# Helper function for saving image files and displaying them
def print_img(title: str, img, show_type: int = NONE, normalize: bool = False, save_image: bool = True):

    if save_image:
        if title is None:
            file_name = TEST_IMG + JPG
        else:
            file_name = TEST_IMG + "_" + title.lower().replace(" ", "_") + JPG

        file_path = os.path.join(OUTPUT_DIRECTORY, file_name)
        print("file_path", file_path)
        cv.imwrite(file_path, img)

    if title is None:
        image_title = TEST_IMG + JPG
    else:
        image_title = TEST_IMG + ": " + title + JPG

    if show_type is OPENCV:
        cv.imshow(image_title, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif show_type is PYPLOT:
        if normalize is True:
            img = img - np.min(img)
            img = img * 255.0 / np.max(img)

        plot.figure()
        plot.title(title)
        plot.imshow(img, cmap='gray')
        plot.waitforbuttonpress()


# ======================================================================================================================
# Main Code
# ======================================================================================================================

test_img = cv.imread(TEST_IMG + BMP)
test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
print_img(TEST_IMG + BMP, test_img)
height, width = test_img.shape

# Calculate the Sobel Gradient Edge.
sobelKernelH = np.array([
    [-1.0, -2.0, -1.0],
    [0.0,   0.0,  0.0],
    [1.0,   2.0,  1.0]
])
sobelKernelV = sobelKernelH.T

sobelImgH = convolve2d(test_img, sobelKernelH, mode='same', boundary='symm', fillvalue=0)
sobelImgV = convolve2d(test_img, sobelKernelV, mode='same', boundary='symm', fillvalue=0)
sobelImgGrad = np.sqrt(np.square(sobelImgH) + np.square(sobelImgV))
print_img("Sobel Image Gradient", sobelImgGrad, normalize=True)

# Create a kernel consisting of 1s and 0s with 1s in the shape of a 35-45px ring.
ringKernel = np.zeros((128, 128))
for y in range(0, 128):
    for x in range(0, 128):
        dist = math.sqrt(((y - 64) ** 2) + ((x - 64) ** 2))  # Distance to center of the kernel

        if 35.0 <= dist <= 45.0:  # Fill with 1s a ring between 35 and 45 pixels from the center.
            ringKernel[x, y] = 1.0
print_img("Ring Kernel", ringKernel, normalize=True)

# Convolve the ring kernel and iris image.
irisRingImg = convolve2d(sobelImgGrad, ringKernel, mode='same', boundary='symm', fillvalue=0)
irisRingImg = irisRingImg * 1 / np.sum(ringKernel)
print_img("Ring Convolution", irisRingImg, normalize=True)

# Apply thresholding to remove all but the most important features.
for y in range(0, height):
    for x in range(0, width):
        if irisRingImg[y, x] < 80:
            irisRingImg[y, x] = 0
print_img("Thresholding", irisRingImg, normalize=True)

# Identify the center of the pupil on the original image/
for index_x, x in enumerate(irisRingImg):
    for index_px, px in enumerate(x):
        if px[SAT] < normal_sat(90):
            px[SAT] = normal_sat(90)
        if px[VAL] > normal_val(30):
            px[HUE] = normal_hue(300)
            px[VAL] = normal_val(35)
            final_img[index_x, index_px] = px

# The bright point of this final image represents the location of the center of the pupil in the image.
