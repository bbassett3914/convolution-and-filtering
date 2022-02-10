"""

@author: Brendan Bassett
@Date: 12/17/2021

CS3150 Image Processing
Dr. Feng Jiang
Metropolitan State University of Denver

Convert a poor-quality image of sheet music to black and white. Then identify the staffs.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================================================================
# SETUP
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

JPG = ".jpg"

IMG_LOC_1 = "Sing We Now Of Christmas - God Rest Ye Merry Gentlemen\\[1]"
IMG_LOC_2 = "Sing We Now Of Christmas - God Rest Ye Merry Gentlemen\\[2]"
IMG_LOC_3 = "Sing We Now Of Christmas - God Rest Ye Merry Gentlemen\\[3]"
IMG_LOC_4 = "Sing We Now Of Christmas - God Rest Ye Merry Gentlemen\\[4]"
IMG_LOC_5 = "Sing We Now Of Christmas - God Rest Ye Merry Gentlemen\\[5]"
IMG_LOC_6 = "Sing We Now Of Christmas - God Rest Ye Merry Gentlemen\\[6]"

OUTPUT = "C:\\Users\\bbass\\PycharmProjects\\OMR\\output\\"

TITLE_1 = "Pg1 "
TITLE_2 = "Pg2 "
TITLE_3 = "Pg3 "
TITLE_4 = "Pg4 "
TITLE_5 = "Pg5 "
TITLE_6 = "Pg6 "

SCALE = 0.8  # fits entire page on my screen

THRESH = 12

HUE = 0
SAT = 1
VAL = 2


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

# Helper for saving image files and displaying them
def print_img(title: str, image, save_location, scale=SCALE):
    bgr_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    bgr_image = cv.resize(bgr_image, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

    if save_location is None:
        cv.imshow(title, bgr_image)
    else:
        cv.imwrite(save_location + title.lower().replace(" ", "_") + JPG, bgr_image)
        cv.imshow(title + JPG + ": ", bgr_image)

    cv.waitKey(0)
    cv.destroyAllWindows()


# Helper for saving image files and displaying them
def plot_img(title: str, image, save_location):
    bgr_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if save_location is not None:
        cv.imwrite(save_location + title.lower().replace(" ", "_") + JPG, bgr_image)

    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.waitforbuttonpress()


# Applies a simple psudo-black and white threshold to the entire image.
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
            hist = cv.calcHist([block], [0], None, [255], [0, 255])
            plt.plot(hist)
            plt.show(block=True)

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
    print(median_line_spacing)

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


# load and display sample image
gray_img = cv.imread(IMG_LOC_1 + JPG)
gray_img = cv.cvtColor(gray_img, cv.COLOR_BGR2GRAY)
img_height, img_width = gray_img.shape[:2]
print_img(TITLE_1, gray_img, save_location=None)

gray_img = cv.GaussianBlur(gray_img, (3, 3), 0, borderType=cv.BORDER_REPLICATE)

thresh_img = b_w_thresh_grad(gray_img, 64)

# apply morphological opening to remove some small artifacts
avgKernel = np.ones((2, 2), np.uint8)
open_img = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, avgKernel)

# Threshold to full black and white
thresh2_img = b_w_thresh(open_img)

adapt_img = np.zeros((img_height, img_width), np.uint8)
cv.threshold(gray_img, 120, 255, cv.THRESH_OTSU, dst=adapt_img)
cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 10, dst=adapt_img)
cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10, dst=adapt_img)

line_spacing2, staffs_img = detect_staffs(adapt_img, (2, 3), (9, 12))

print_img(TITLE_1 + "staff detection with sets", staffs_img, save_location=OUTPUT)
