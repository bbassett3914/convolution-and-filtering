"""

@author: Brendan Bassett
@Date: 12/17/2021

CS3150 Image Processing
Dr. Feng Jiang
Metropolitan State University of Denver

Convert a poor-quality image of sheet music to black and white. Then identify the staff lines.
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================================================================
# SETUP
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

TEST_IMG = "test"
JPG = ".jpg"
BMP = ".bmp"

ROOT_DIRECTORY = os.path.abspath('.')
print("ROOT DIRECTORY", ROOT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.abspath(os.path.join("output"))
print("OUTPUT DIRECTORY", OUTPUT_DIRECTORY)

NONE = 0
OPENCV = 1
PYPLOT = 2

SCALE = 0.8  # fits entire page on my screen

THRESH = 12

HUE = 0
SAT = 1
VAL = 2


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

# Helper function for saving image files and displaying them.
def print_img(title: str, img, show_type: int = NONE, normalize: bool = False, save_image: bool = True):

    bgr_image = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    if save_image:
        if title is None:
            file_name = TEST_IMG + JPG
        else:
            file_name = TEST_IMG + "_" + title.lower().replace(" ", "_") + JPG

        file_path = os.path.join(OUTPUT_DIRECTORY, file_name)
        cv.imwrite(file_path, bgr_image)

    if title is None:
        image_title = TEST_IMG + JPG
    else:
        image_title = TEST_IMG + ": " + title + JPG

    if show_type is OPENCV:
        cv.imshow(image_title, bgr_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif show_type is PYPLOT:
        if normalize is True:
            img = img - np.min(bgr_image)
            img = img * 255.0 / np.max(bgr_image)

        plt.figure()
        plt.title(TEST_IMG + JPG + ": " + title)
        plt.imshow(bgr_image)
        plt.waitforbuttonpress()


# Applies a simple pseudo-black and white threshold to the entire image.
def b_w_thresh(img, ratio=0):
    height, width = img.shape[:2]
    col_avg = img.mean(axis=0)
    px_avg = col_avg.mean(axis=0)

    # cv.threshold(img, px_avg, 255, cv.THRESH_BINARY_INV)

    for x in range(width):
        for y in range(height):
            if img[y][x] < px_avg - THRESH:
                img[y][x] = ratio

    return img


# Applies a pseudo-black and white threshold to smaller blocks, or parts, of the image. This helps to detect b&w text
# when part of the page is dark overall and another part is lighter.
def b_w_thresh_grad(img, block_size: int):

    height, width = img.shape[:2]

    # calculate the extra margin left after partitioning into full [block_size] by [block_size] sections. This will be
    # added to the right and bottom side blocks
    extra_x = width % block_size
    extra_y = height % block_size

    # iterate through each of the blocks
    h_range = range(height // block_size)
    for yi in h_range:

        w_range = range(width // block_size)
        for xi in w_range:

            # identify the exact location of the block within the full image
            l = xi * block_size
            if xi < len(w_range) - 1:
                r = xi * block_size + block_size - 1
            else:                   # add on the extra right-side margin to ensure the full image is processed
                r = xi * block_size + block_size - 1 + extra_x

            t = yi * block_size
            if yi < len(h_range) - 1:
                b = yi * block_size + block_size - 1
            else:                   # add on the extra bottom-side margin to ensure the full image is processed
                b = yi * block_size + block_size - 1 + extra_y

            block = img[t:b, l:r]
            block_title = "block " + str(xi) + "," + str(yi)
            # block_dimensions = block_title + "::  l==" + str(l) + " r==" + str(r) + " t==" + str(t) + " b==" + str(b)
            # print(block_dimensions)

            if len(block) == 0:
                print("ERROR: This function attempted to execute on a nonexistent block. /n" + block_title)
                continue

            # display of the histogram of intensity values within this block
            # hist = cv.calcHist([block], [0], None, [255], [0, 255])
            # plt.plot(hist)
            # plt.show(block=True)

            # obtain the mean intensity value within the block
            col_avg = block.mean(axis=0)
            px_avg = col_avg.mean(axis=0)

            # apply a binarization threshold around the mean intensity value of the block
            for x in range(l, r + 1):
                for y in range(t, b + 1):

                    if img[y][x] >= px_avg - THRESH:
                        img[y][x] = 255
                    else:       # this is not full binarization, as darker colors are simply made darker, rather than
                                # being made fully black.
                        img[y][x] = img[y][x] // 3

    return img


# A simple staff line detection algorithm
def detect_staffs(img, line_thickness: tuple, spacing: tuple):

    line_img = np.zeros((img_height, img_width), np.uint8)
    line_spacing_list = [] # a place to record the distance between staff lines whenever two or more are detected.

    # Show where there are vertical runs of black pixels in a row with the given range
    for x, col in enumerate(img.T):
        run_count = 0

        for y, val in enumerate(col):

            if val < 50:
                run_count += 1
            else:
                if line_thickness[0] <= run_count <= line_thickness[1]:
                    for run_back in range(run_count):
                        line_img[y-run_back][x] = 255
                run_count = 0

    line_spacing_img = np.zeros((img_height, img_width), np.uint8)

    # Show where there are vertical runs of white pixels in between the identified black pixel runs.
    for x, col in enumerate(line_img.T):
        run_count = 0\

        for y, val in enumerate(col):

            if val < 50:
                run_count += 1

            else:
                if spacing[0] <= run_count <= spacing[1]:
                    line_spacing_list.append(run_count)

                    for line in range(max(line_thickness)):
                        if line_img[y-run_count-line][x] == 255:
                            line_spacing_img[y-run_count-line][x] = 255
                        if line_img[y+line][x] == 255:
                            line_spacing_img[y+line][x] = 255

                run_count = 0

    line_sets_img = np.zeros((img_height, img_width), np.uint8)
    median_line_spacing = np.median(line_spacing_list)
    print("     median line spacing:", median_line_spacing)

    # Show where there are several parallel lines forming a staff
    for x, col in enumerate(line_img.T):
        space_count = 0
        set_count = 0

        for y, val in enumerate(col):

            if val > 200:
                if spacing[0] <= space_count <= spacing[1]:  # another staff line is identified
                    set_count += 1                                # add this staff line to the number of sets of lines

                    if set_count == 2:     # 3 lines in this set. Go back and mark the first two lines as well.
                        for run_back in range((max(line_thickness) * 2) + (round(median_line_spacing) * 2)):
                            if line_spacing_img[y-run_back][x] == 255:
                                line_sets_img[y-run_back][x] = 255
                        for run_ahead in range(max(line_thickness)):
                            if line_spacing_img[y+run_ahead][x] == 255:
                                line_sets_img[y+run_ahead][x] = 255

                    elif set_count > 2:   # this is not the first set. Just mark this line.
                        for run_ahead in range(max(line_thickness)):
                            if line_spacing_img[y+run_ahead][x] == 255:
                                line_sets_img[y+run_ahead][x] = 255

                        if set_count == 4:  # this must be the last line of the set. Start the set over
                            set_count = 0

                elif space_count != 0:       # invalid space length. This ends the set of lines if there is one
                    set_count = 0

                space_count = 0

            else:
                space_count += 1

    return median_line_spacing, line_sets_img


# ======================================================================================================================
# MAIN CODE
# ======================================================================================================================

print("Loading and preprocessing the image...")

# load and display sample image
test_img = cv.imread(TEST_IMG + JPG)
gray_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
img_height, img_width = gray_img.shape[:2]

# ----------------------------------------------------------------------------------------------------------------------
# Binarization Method 1 - Thresholding in Steps
# ----------------------------------------------------------------------------------------------------------------------

# Apply some simple preprocessing to make later steps more consistent.

gray_img = cv.GaussianBlur(gray_img, (3, 3), 0, borderType=cv.BORDER_REPLICATE)
thresh_img = b_w_thresh_grad(gray_img, 64)

# apply morphological opening to remove some small artifacts
avgKernel = np.ones((2, 2), np.uint8)
open_img = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, avgKernel)

print_img("Preprocessing Complete", open_img)

print("Binarizing the image into black and white...")

# Threshold to full binary black and white.
thresh_steps_img = b_w_thresh(open_img)

print_img("Thresholding by Steps", thresh_steps_img)

print("Applying adaptive thresholding...")

# ----------------------------------------------------------------------------------------------------------------------
# Binarization Method 2 - Adaptive Thresholding
# ----------------------------------------------------------------------------------------------------------------------

thresh_adapt_mean_img = np.zeros((img_height, img_width), np.uint8)
cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 10, dst=thresh_adapt_mean_img)

thresh_adapt_gauss_img = np.zeros((img_height, img_width), np.uint8)
cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 19, 10, dst=thresh_adapt_gauss_img)

print_img("Adaptive Thresholding by Mean", thresh_adapt_mean_img)
print_img("Adaptive Thresholding by Gaussian", thresh_adapt_gauss_img)

# ----------------------------------------------------------------------------------------------------------------------
# Staff Line Detection
# ----------------------------------------------------------------------------------------------------------------------

print("Detecting staff lines...")
line_spacing, staffs_detect_img = detect_staffs(thresh_steps_img, (2, 3), (9, 12))

print_img("Final Result (Inverted)", staffs_detect_img)

invt_staffs_detect_img = cv.bitwise_not(staffs_detect_img)

print_img("Final Result", invt_staffs_detect_img)
