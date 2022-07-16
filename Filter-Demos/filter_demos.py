"""
Created on Wed Sep 29 09:50:06 2021

@author: Brendan Bassett
@Date: 10/6/2021

Written for CS3150 Image Processing
Dr. Feng Jiang
Metropolitan State University of Denver
"""

import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plot
from scipy.signal import convolve2d

# ======================================================================================================================
# Setup
# ======================================================================================================================

TEST_IMG = "RezzSelfie"
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

imgC = cv.imread(TEST_IMG + JPG)
imgG = cv.cvtColor(imgC, cv.COLOR_BGR2GRAY)
height, width, layers = imgC.shape

# Check that loaded images display correctly
print_img("Color", imgC, save_image=False)
print_img("Grayscale", imgG)

# ----------------------------------------------------------------------------------------------------------------------
# Averaging Filters
# ----------------------------------------------------------------------------------------------------------------------

print("Averaging Filters...")

avgKernel = np.full((3, 3), 1 / 9)
avgImg = cv.filter2D(imgC, -1, avgKernel)
print_img("3x3 Average", avgImg)

avgKernel = np.full((5, 5), 1 / 25)
avgImg = cv.filter2D(imgC, -1, avgKernel)
print_img("5x5 Average", avgImg)

avgKernel = np.full((9, 9), 1 / 81)
avgImg = cv.filter2D(imgC, -1, avgKernel)
print_img("9x9 Average", avgImg)

avgKernel = np.full((13, 13), 1 / 169)
avgImg = cv.filter2D(imgC, -1, avgKernel)
print_img("13x13 Average", avgImg)

# ----------------------------------------------------------------------------------------------------------------------
# Sobel Filters
# ----------------------------------------------------------------------------------------------------------------------

print("Sobel Filters...")

sobelKernelH = np.array([
    [-1.0, -2.0, -1.0],
    [0.0,   0.0,  0.0],
    [1.0,   2.0,  1.0]
])

sobelH = convolve2d(imgG, sobelKernelH, mode='same', boundary='symm', fillvalue=0)
print_img("Horizontal Sobel", sobelH, normalize=True)

sobelKernelV = sobelKernelH.T
sobelV = convolve2d(imgG, sobelKernelV, mode='same', boundary='symm', fillvalue=0)
print_img("Vertical Sobel", sobelV, normalize=True)

sobelImgGrad = np.sqrt(np.square(sobelH) + np.square(sobelV))
print_img("Gradient Sobel", sobelImgGrad, normalize=True)

# ----------------------------------------------------------------------------------------------------------------------
# Laplacian Filters
# ----------------------------------------------------------------------------------------------------------------------

print("Laplacian Filters...")

laplacianKernel1 = np.array([
    [0.0,  1.0, 0.0],
    [1.0, -4.0, 1.0],
    [0.0,  1.0, 0.0]
])

laplacian1 = convolve2d(imgG, laplacianKernel1, mode='same', boundary='symm', fillvalue=0)
print_img("3x3 Laplacian #1", laplacian1, normalize=True)

laplacianKernel2 = np.array([
    [1.0,  1.0, 1.0],
    [1.0, -8.0, 1.0],
    [1.0,  1.0, 1.0]
])

laplacian2 = convolve2d(imgG, laplacianKernel2, mode='same', boundary='symm', fillvalue=0)
print_img("3x3 Laplacian #2", laplacian2, normalize=True)

# ----------------------------------------------------------------------------------------------------------------------
# Median Filters
# ----------------------------------------------------------------------------------------------------------------------

print("Median Filters...")

medianSM = np.zeros((height, width), dtype=float)
for y in range(1, height - 2):
    for x in range(1, width - 2):
        neighborhood = imgG[y - 1:y + 2, x - 1:x + 2]
        sorted_pixels = sorted(np.ndarray.flatten(neighborhood))
        medianSM[y][x] = sorted_pixels[4]
print_img("3x3 Median Filter", medianSM)

medianMD = np.zeros((height, width), dtype=float)
for y in range(2, height - 3):
    for x in range(2, width - 3):
        neighborhood = imgG[y - 2:y + 3, x - 2:x + 3]
        sorted_pixels = sorted(np.ndarray.flatten(neighborhood))
        medianMD[y][x] = sorted_pixels[12]
print_img("5x5 Median Filter", medianMD)

medianLG = np.zeros((height, width), dtype=float)
for y in range(3, height - 4):
    for x in range(3, width - 4):
        neighborhood = imgG[y - 3:y + 4, x - 3:x + 4]
        sorted_pixels = sorted(np.ndarray.flatten(neighborhood))
        medianLG[y][x] = sorted_pixels[24]
print_img("7x7 Median Filter", medianLG)

# ----------------------------------------------------------------------------------------------------------------------
# Gaussian Filters
# ----------------------------------------------------------------------------------------------------------------------

print("Gaussian Filters...")

gaussianKernelSM = np.array([
    [1.0,  2.0, 1.0],
    [2.0,  4.0, 2.0],
    [1.0,  2.0, 1.0]
])
gaussianSM = convolve2d(imgG, gaussianKernelSM, mode='same', boundary='symm', fillvalue=0)
gaussianSM = gaussianSM/np.sum(gaussianKernelSM)
print_img("3x3 Gaussian", gaussianSM)

gaussianKernelMD1 = np.array([
    [2.0,  7.0,  12.0,  7.0,  2.0],
    [7.0,  31.0, 52.0,  31.0, 7.0],
    [12.0, 52.0, 127.0, 52.0, 12.0],
    [7.0,  31.0, 52.0,  31.0, 7.0],
    [2.0,  7.0,  12.0,  7.0,  2.0]
])
gaussianMD1 = convolve2d(imgG, gaussianKernelMD1, mode='same', boundary='symm', fillvalue=0)
gaussianMD1 = gaussianMD1/np.sum(gaussianKernelMD1)
print_img("5x5 Gaussian #1", gaussianMD1)

gaussianKernelMD2 = np.array([
    [1.0, 4.0,  7.0,  4.0,  1.0],
    [4.0, 16.0, 26.0, 16.0, 4.0],
    [7.0, 26.0, 41.0, 26.0, 7.0],
    [4.0, 16.0, 26.0, 16.0, 4.0],
    [1.0, 4.0,  7.0,  4.0,  1.0]
])
gaussianMD2 = convolve2d(imgG, gaussianKernelMD2, mode='same', boundary='symm', fillvalue=0)
gaussianMD2 = gaussianMD2/np.sum(gaussianKernelMD2)
print_img("5x5 Gaussian #2", gaussianMD2)

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
print_img("7x7 Gaussian", gaussianLG)

# ----------------------------------------------------------------------------------------------------------------------
# Central Difference Filters
# ----------------------------------------------------------------------------------------------------------------------

print("Central Difference Filters...")

cdKernelV = np.array([[1.0, 0.0, -1.0]])
cdV = convolve2d(imgG, cdKernelV, mode='same', boundary='symm', fillvalue=0)
print_img("Vertical Central Difference", cdV, normalize=True)

cdKernelH = cdKernelV.T
cdH = convolve2d(imgG, cdKernelH, mode='same', boundary='symm', fillvalue=0)
print_img("Horizontal Central Difference", cdH, normalize=True)

# ----------------------------------------------------------------------------------------------------------------------
# Prewitt Filters
# ----------------------------------------------------------------------------------------------------------------------

print("Prewitt Filters...")

prewittKernelV = np.array([
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0]
])
prewittV = convolve2d(imgG, prewittKernelV, mode='same', boundary='symm', fillvalue=0)
print_img("Vertical Prewitt", prewittV, normalize=True)

prewittKernelH = prewittKernelV.T
prewittH = convolve2d(imgG, prewittKernelH, mode='same', boundary='symm', fillvalue=0)
print_img("Horizontal Prewitt", prewittH, normalize=True)

# ----------------------------------------------------------------------------------------------------------------------
# Laplacian of Gaussian Filters
# ----------------------------------------------------------------------------------------------------------------------

print("Laplacian of Gaussian Filters...")

logKernelSM = np.array([
    [0.0, 0.0,  1.0,  0.0, 0.0],
    [0.0, 1.0,  2.0,  1.0, 0.0],
    [1.0, 2.0, -16.0, 2.0, 1.0],
    [0.0, 1.0,  2.0,  1.0, 0.0],
    [0.0, 0.0,  1.0,  0.0, 0.0]
])
logSM = convolve2d(imgG, logKernelSM, mode='same', boundary='symm', fillvalue=0)
print_img("5x5 Laplacian of Gaussian", logSM, normalize=True)

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
print_img("7x7 Laplacian of Gaussian", logMD, normalize=True)

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
print_img("9x9 Laplacian of Gaussian", logLG, normalize=True)
