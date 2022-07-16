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

REZZ_IMG = "RezzSelfie"
IRIS_IMG = "iris-1"
JPG = ".jpg"
BMP = ".bmp"

ROOT_DIRECTORY = os.path.abspath('.')
print("ROOT DIRECTORY", ROOT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.abspath(os.path.join("output_images"))
print("OUTPUT DIRECTORY", OUTPUT_DIRECTORY)

NONE = 0
OPENCV = 1
PYPLOT = 2

imgC = cv.imread(REZZ_IMG + JPG)
imgG = cv.cvtColor(imgC, cv.COLOR_BGR2GRAY)
height, width, layers = imgC.shape


# Helper function for saving image files and displaying them
def print_img(title: str, img, show_type: int = NONE, normalize: bool = False):

    if title is None:
        file_name = REZZ_IMG + JPG
    else:
        file_name = REZZ_IMG + " " + title + JPG

    file_path = os.path.join(OUTPUT_DIRECTORY, file_name)
    cv.imwrite(file_path, img)

    if title is None:
        image_title = REZZ_IMG + JPG
    else:
        image_title = REZZ_IMG + " : " + title + JPG

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


# Check that loaded images display correctly
print_img("Color", imgC, OPENCV, False)
print_img("Grayscale", imgG, OPENCV, False)


# ======================================================================================================================
# 1. Iris Detection
# ======================================================================================================================

irisImg = cv.imread(IRIS_IMG + BMP)
irisImg = cv.cvtColor(irisImg, cv.COLOR_BGR2GRAY)
print_img(IRIS_IMG + BMP, irisImg, show_type=OPENCV)
height, width = irisImg.shape

# Calculate the Sobel Gradient Edge.
sobelKernelH = np.array([
    [-1.0, -2.0, -1.0],
    [0.0,   0.0,  0.0],
    [1.0,   2.0,  1.0]
])
sobelKernelV = sobelKernelH.T

sobelImgH = convolve2d(irisImg, sobelKernelH, mode='same', boundary='symm', fillvalue=0)
sobelImgV = convolve2d(irisImg, sobelKernelV, mode='same', boundary='symm', fillvalue=0)
sobelImgGrad = np.sqrt(np.square(sobelImgH) + np.square(sobelImgV))
print_img("Iris: Sobel Image Gradient", sobelImgGrad, PYPLOT, True)

# Create a kernel consisting of 1s and 0s with 1s in the shape of a 35-45px ring
ringKernel = np.zeros((128, 128))
for y in range(0, 128):
    for x in range(0, 128):
        dist = math.sqrt(((y - 64) ** 2) + ((x - 64) ** 2))  # Distance to center of the kernel

        if 35.0 <= dist <= 45.0:  # Fill with 1s a ring between 35 and 45 pixels from the center.
            ringKernel[x, y] = 1.0
print_img("Ring Kernel", ringKernel, PYPLOT, True)

# Convolve the ring kernel and iris image
irisRingImg = convolve2d(sobelImgGrad, ringKernel, mode='same', boundary='symm', fillvalue=0)
irisRingImg = irisRingImg * 1 / np.sum(ringKernel)
print_img("Iris: Ring Convolution", irisRingImg, PYPLOT, True)

# Apply thresholding to remove all but the most important features
for y in range(0, height):
    for x in range(0, width):
        if irisRingImg[y, x] < 80:
            irisRingImg[y, x] = 0
print_img("Iris: Thresholding", irisRingImg, PYPLOT, True)

# The bright point of this final image represents the location of the center of the pupil in the image.

# ======================================================================================================================
# 2. FILTER DEMONSTRATIONS
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# Averaging Filters
# ----------------------------------------------------------------------------------------------------------------------

avgKernel = np.full((3, 3), 1 / 9)
avgImg = cv.filter2D(imgC, -1, avgKernel)
print_img("3x3 Average", avgImg, OPENCV, False)

avgKernel = np.full((5, 5), 1 / 25)
avgImg = cv.filter2D(imgC, -1, avgKernel)
print_img("5x5 Average", avgImg, OPENCV, False)

avgKernel = np.full((9, 9), 1 / 81)
avgImg = cv.filter2D(imgC, -1, avgKernel)
print_img("9x9 Average", avgImg, OPENCV, False)

avgKernel = np.full((13, 13), 1 / 169)
avgImg = cv.filter2D(imgC, -1, avgKernel)
print_img("13x13 Average", avgImg, OPENCV, False)

# ----------------------------------------------------------------------------------------------------------------------
# Sobel Filters
# ----------------------------------------------------------------------------------------------------------------------

sobelKernelH = np.array([
    [-1.0, -2.0, -1.0],
    [0.0,   0.0,  0.0],
    [1.0,   2.0,  1.0]
])

sobelH = convolve2d(imgG, sobelKernelH, mode='same', boundary='symm', fillvalue=0)
print_img("Horizontal Sobel", sobelH, PYPLOT, True)

sobelKernelV = sobelKernelH.T
sobelV = convolve2d(imgG, sobelKernelV, mode='same', boundary='symm', fillvalue=0)
print_img("Vertical Sobel", sobelV, PYPLOT, True)

sobelImgGrad = np.sqrt(np.square(sobelH) + np.square(sobelV))
print_img("Gradient Sobel", sobelImgGrad, PYPLOT, True)

# ----------------------------------------------------------------------------------------------------------------------
# Laplacian Filters
# ----------------------------------------------------------------------------------------------------------------------

laplacianKernel1 = np.array([
    [0.0,  1.0, 0.0],
    [1.0, -4.0, 1.0],
    [0.0,  1.0, 0.0]
])

laplacian1 = convolve2d(imgG, laplacianKernel1, mode='same', boundary='symm', fillvalue=0)
print_img("3x3 Laplacian #1", laplacian1, PYPLOT, True)

laplacianKernel2 = np.array([
    [1.0,  1.0, 1.0],
    [1.0, -8.0, 1.0],
    [1.0,  1.0, 1.0]
])

laplacian2 = convolve2d(imgG, laplacianKernel2, mode='same', boundary='symm', fillvalue=0)
print_img("3x3 Laplacian #2", laplacian2, PYPLOT, True)

# ----------------------------------------------------------------------------------------------------------------------
# Median Filters
# ----------------------------------------------------------------------------------------------------------------------

medianSM = np.zeros((height, width), dtype=float)
for y in range(1, height - 2):
    for x in range(1, width - 2):
        neighborhood = imgG[y - 1:y + 2, x - 1:x + 2]
        sorted_pixels = sorted(np.ndarray.flatten(neighborhood))
        medianSM[y][x] = sorted_pixels[4]
print_img("3x3 Median Filter", medianSM, PYPLOT, False)

medianMD = np.zeros((height, width), dtype=float)
for y in range(2, height - 3):
    for x in range(2, width - 3):
        neighborhood = imgG[y - 2:y + 3, x - 2:x + 3]
        sorted_pixels = sorted(np.ndarray.flatten(neighborhood))
        medianMD[y][x] = sorted_pixels[12]
print_img("5x5 Median Filter", medianMD, PYPLOT, False)

medianLG = np.zeros((height, width), dtype=float)
for y in range(3, height - 4):
    for x in range(3, width - 4):
        neighborhood = imgG[y - 3:y + 4, x - 3:x + 4]
        sorted_pixels = sorted(np.ndarray.flatten(neighborhood))
        medianLG[y][x] = sorted_pixels[24]
print_img("7x7 Median Filter", medianLG, PYPLOT, False)

# ----------------------------------------------------------------------------------------------------------------------
# Gaussian Filters
# ----------------------------------------------------------------------------------------------------------------------

gaussianKernelSM = np.array([
    [1.0,  2.0, 1.0],
    [2.0,  4.0, 2.0],
    [1.0,  2.0, 1.0]
])
gaussianSM = convolve2d(imgG, gaussianKernelSM, mode='same', boundary='symm', fillvalue=0)
gaussianSM = gaussianSM/np.sum(gaussianKernelSM)
print_img("3x3 Gaussian", gaussianSM, PYPLOT, False)

gaussianKernelMD1 = np.array([
    [2.0,  7.0,  12.0,  7.0,  2.0],
    [7.0,  31.0, 52.0,  31.0, 7.0],
    [12.0, 52.0, 127.0, 52.0, 12.0],
    [7.0,  31.0, 52.0,  31.0, 7.0],
    [2.0,  7.0,  12.0,  7.0,  2.0]
])
gaussianMD1 = convolve2d(imgG, gaussianKernelMD1, mode='same', boundary='symm', fillvalue=0)
gaussianMD1 = gaussianMD1/np.sum(gaussianKernelMD1)
print_img("5x5 Gaussian #1", gaussianMD1, PYPLOT, False)

gaussianKernelMD2 = np.array([
    [1.0, 4.0,  7.0,  4.0,  1.0],
    [4.0, 16.0, 26.0, 16.0, 4.0],
    [7.0, 26.0, 41.0, 26.0, 7.0],
    [4.0, 16.0, 26.0, 16.0, 4.0],
    [1.0, 4.0,  7.0,  4.0,  1.0]
])
gaussianMD2 = convolve2d(imgG, gaussianKernelMD2, mode='same', boundary='symm', fillvalue=0)
gaussianMD2 = gaussianMD2/np.sum(gaussianKernelMD2)
print_img("5x5 Gaussian #2", gaussianMD2, PYPLOT, False)

gaussianKernelLG = np.array([
    [1.0, 1.0, 2.0, 2.0,  2.0, 1.0, 1.0],
    [1.0, 3.0, 4.0, 5.0,  4.0, 3.0, 1.0],
    [2.0, 4.0, 7.0, 8.0,  7.0, 4.0, 2.0],
    [2.0, 5.0, 8.0, 10.0, 8.0, 5.0, 2.0],
    [2.0, 4.0, 7.0, 8.0,  7.0, 4.0, 2.0],
    [1.0, 3.0, 4.0, 5.0,  4.0, 3.0, 1.0],
    [1.0, 1.0, 2.0, 2.0,  2.0, 1.0, 1.0]
])
gaussianLG = convolve2d(imgG, gaussianKernelLG, mode='same', boundary='symm', fillvalue=0)
gaussianLG = gaussianLG/np.sum(gaussianKernelLG)
print_img("7x7 Gaussian", gaussianLG, PYPLOT, False)

# ----------------------------------------------------------------------------------------------------------------------
# Central Difference Filters
# ----------------------------------------------------------------------------------------------------------------------

cdKernelV = np.array([[1.0, 0.0, -1.0]])
cdV = convolve2d(imgG, cdKernelV, mode='same', boundary='symm', fillvalue=0)
print_img("Vertical Central Difference", cdV, PYPLOT, True)

cdKernelH = cdKernelV.T
cdH = convolve2d(imgG, cdKernelH, mode='same', boundary='symm', fillvalue=0)
print_img("Horizontal Central Difference", cdH, PYPLOT, True)

# ----------------------------------------------------------------------------------------------------------------------
# Prewitt Filters
# ----------------------------------------------------------------------------------------------------------------------

prewittKernelV = np.array([
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0]
])
prewittV = convolve2d(imgG, prewittKernelV, mode='same', boundary='symm', fillvalue=0)
print_img("Vertical Prewitt", prewittV, PYPLOT, True)

prewittKernelH = prewittKernelV.T
prewittH = convolve2d(imgG, prewittKernelH, mode='same', boundary='symm', fillvalue=0)
print_img("Horizontal Prewitt", prewittH, PYPLOT, True)

# ----------------------------------------------------------------------------------------------------------------------
# Laplacian of Gaussian Filters
# ----------------------------------------------------------------------------------------------------------------------

logKernelSM = np.array([
    [0.0, 0.0,  1.0,  0.0, 0.0],
    [0.0, 1.0,  2.0,  1.0, 0.0],
    [1.0, 2.0, -16.0, 2.0, 1.0],
    [0.0, 1.0,  2.0,  1.0, 0.0],
    [0.0, 0.0,  1.0,  0.0, 0.0]
])
logSM = convolve2d(imgG, logKernelSM, mode='same', boundary='symm', fillvalue=0)
print_img("5x5 Laplacian of Gaussian", logSM, PYPLOT, True)

logKernelMD = np.array([
    [0.0, 0.0,  1.0,  1.0,   1.0, 0.0, 0.0],
    [0.0, 1.0,  3.0,  3.0,   3.0, 1.0, 0.0],
    [1.0, 3.0,  0.0, -7.0,   0.0, 3.0, 1.0],
    [1.0, 3.0, -7.0, -24.0, -7.0, 3.0, 1.0],
    [1.0, 3.0,  0.0, -7.0,   0.0, 3.0, 1.0],
    [0.0, 1.0,  3.0,  3.0,   3.0, 1.0, 0.0],
    [0.0, 0.0,  1.0,  1.0,   1.0, 0.0, 0.0]
])
logMD = convolve2d(imgG, logKernelMD, mode='same', boundary='symm', fillvalue=0)
print_img("7x7 Laplacian of Gaussian", logMD, PYPLOT, True)

logKernelLG = np.array([
    [0.0,  1.0, 1.0,  2.0,   2.0,   2.0,  1.0,  1.0, 0.0],
    [1.0,  2.0, 4.0,  5.0,   5.0,   5.0,  4.0,  2.0, 1.0],
    [1.0,  4.0, 5.0,  3.0,   0.0,   3.0,  5.0,  4.0, 1.0],
    [2.0,  5.0, 3.0, -12.0, -24.0, -12.0, 3.0,  5.0, 2.0],
    [2.0,  5.0, 0.0, -24.0, -40.0, -24.0, 0.0,  5.0, 2.0],
    [2.0,  5.0, 3.0, -12.0, -24.0, -12.0, 3.0,  5.0, 2.0],
    [1.0,  4.0, 5.0,  3.0,   0.0,   3.0,  5.0,  4.0, 1.0],
    [1.0,  2.0, 4.0,  5.0,   5.0,   5.0,  4.0,  2.0, 1.0],
    [0.0,  1.0, 1.0,  2.0,   2.0,   2.0,  1.0,  1.0, 0.0]
])

logLG = convolve2d(imgG, logKernelLG, mode='same', boundary='symm', fillvalue=0)
print_img("9x9 Laplacian of Gaussian", logLG, PYPLOT, True)
