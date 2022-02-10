"""

@author: Brendan Bassett
@Date: 11/14/2021

CS3150 Image Processing
Dr. Feng Jiang
Metropolitan State University of Denver

Detect and distinguish different routes on a climbing gym wall by the color of their holds.
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
TEST_IMG = "test2-1"

OUTPUT_LOCATION = "C:\\Users\\bbass\\PycharmProjects\\Image_Enhancement\\output\\"

NONE = 0
CV = 1
PLT = 2

HUE = 0
SAT = 1
VAL = 2


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

# Helper for saving image files and displaying them
def print_img(title: str, image, show_type: int = PLT, save_image=False):
    bgr_image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
    if save_image:
        cv.imwrite(OUTPUT_LOCATION + TEST_IMG + "." + title.lower().replace(" ", "_") + JPG, bgr_image)

    if show_type is CV:
        cv.imshow(TEST_IMG + JPG + ": " + title, bgr_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif show_type is PLT:
        plt.figure()
        plt.title(TEST_IMG + JPG + ": " + title)
        plt.imshow(image)
        plt.waitforbuttonpress()


# Shows a histogram of the hue, saturation, and value in the image
def hsv_histogram(image):
    # show a histogram of hue in the image
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    hist[0] = 0
    plt.plot(hist)

    # show a histogram of saturation in the image
    hist = cv.calcHist([image], [1], None, [256], [0, 256])
    hist[0] = 0
    plt.plot(hist)

    # show a histogram of value in the image
    hist = cv.calcHist([image], [2], None, [256], [0, 256])
    hist[0] = 0
    plt.plot(hist)
    plt.show(block=True)


# Converts HSV hue values from HTML range [0,359] to opencv range [0,179]
def normal_hue(hue):
    return round(hue * 179.0 / 360.0)


# Converts HSV hue values from HTML range [0,100] to opencv range [0,255]
def normal_sat(saturation):
    return round(saturation * 255.0 / 100.0)


# Converts HSV hue values from HTML range [0,100] to opencv range [0,255]
def normal_val(value):
    return round(value * 255.0 / 100.0)


# Converts HSV values from HTML range to opencv range.
def normal_hsv(hue, saturation, value):
    return [normal_hue(hue), normal_sat(saturation), normal_val(value)]


# ======================================================================================================================
# MAIN CODE
# ======================================================================================================================

# load and display sample image
img = cv.imread(TEST_IMG + JPG)
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
print_img("", img, NONE)

# Desaturate the image as a gray base for the final image. It makes the colors of the identified routes stand out.
img_final = img.copy()
for x in img_final:
    for px in x:
        px[SAT] = 0

# ----------------------------------------------------------------------------------------------------------------------
# Identify Red Holds
# ----------------------------------------------------------------------------------------------------------------------

red_img = img.copy()
red_img = cv.GaussianBlur(red_img, (5, 5), 0)

# remove all but the brightest and most saturated red colors
for x in red_img:
    for px in x:
        if normal_hue(15) < px[HUE] < normal_hue(350) or px[SAT] < normal_sat(30) or px[VAL] < normal_val(50):
            px[VAL] = 0
# print_img("Red Holds", red_img, NONE, True)

# apply morphological opening to remove any small artifacts that are not holds and create a defined edge for holds
avgKernel = np.ones((8, 8), np.uint8)
red_img = cv.morphologyEx(red_img, cv.MORPH_OPEN, avgKernel)
print_img("Red Holds Opening", red_img, NONE, True)

# apply morphological closing to make the holds to fill in chalky areas
avgKernel = np.ones((60, 60), np.uint8)
red_img = cv.morphologyEx(red_img, cv.MORPH_CLOSE, avgKernel)
print_img("Red Holds Closing", red_img, NONE, True)

# Make the holds solid, saturated red for an even texture
for index_x, x in enumerate(red_img):
    for index_px, px in enumerate(x):
        if px[SAT] < normal_sat(90):
            px[SAT] = normal_sat(90)
        if px[VAL] > normal_val(30):
            px[HUE] = normal_hue(359)
            px[VAL] = normal_val(37)
            img_final[index_x, index_px] = px

print_img("Red Holds Solid", red_img, NONE, True)

# Identify the boundaries of each hold and mark them on the original image
red_gray = cv.cvtColor(red_img, cv.COLOR_HSV2BGR)
red_gray = cv.cvtColor(red_gray, cv.COLOR_BGR2GRAY)
contours, hierarchy = cv.findContours(red_gray, 1, 2)
for contour in contours:
    convexHull = cv.convexHull(contour)
    cv.drawContours(img_final, [convexHull], -1, (0, 0, 0), 7)
print_img("Red Holds Contour", img_final, NONE, True)


# ----------------------------------------------------------------------------------------------------------------------
# Identify Pink Holds
# ----------------------------------------------------------------------------------------------------------------------

pink_img = img.copy()
pink_img = cv.GaussianBlur(pink_img, (5, 5), 0)

# remove all but the brightest and most saturated pink colors
for x in pink_img:
    for px in x:
        if px[HUE] < normal_hue(300) or px[HUE] > normal_hue(345) or px[SAT] < normal_sat(30) or px[VAL] < normal_val(50):
            px[VAL] = 0
print_img("Pink Holds", pink_img, NONE, True)

# apply morphological opening to remove any small artifacts that are not holds and create a defined edge for holds
avgKernel = np.ones((8, 8), np.uint8)
pink_img = cv.morphologyEx(pink_img, cv.MORPH_OPEN, avgKernel)
print_img("Pink Holds Opening", pink_img, NONE, True)

# apply morphological closing to make the holds to fill in chalky areas
avgKernel = np.ones((60, 60), np.uint8)
pink_img = cv.morphologyEx(pink_img, cv.MORPH_CLOSE, avgKernel)
print_img("Pink Holds Closing", pink_img, NONE, True)

# Make the holds solid, saturated pink for an even texture
for index_x, x in enumerate(pink_img):
    for index_px, px in enumerate(x):
        if px[SAT] < normal_sat(90):
            px[SAT] = normal_sat(90)
        if px[VAL] > normal_val(30):
            px[HUE] = normal_hue(300)
            px[VAL] = normal_val(35)
            img_final[index_x, index_px] = px

print_img("Pink Holds Solid", pink_img, NONE, True)

# Identify the boundaries of each hold and mark them on the original image
pink_gray = cv.cvtColor(pink_img, cv.COLOR_HSV2BGR)
pink_gray = cv.cvtColor(pink_gray, cv.COLOR_BGR2GRAY)
contours, hierarchy = cv.findContours(pink_gray, 1, 2)
for contour in contours:
    convexHull = cv.convexHull(contour)
    cv.drawContours(img_final, [convexHull], -1, (0, 0, 0), 7)
print_img("Pink Holds Contour", img_final, NONE, True)

# ----------------------------------------------------------------------------------------------------------------------
# Identify Yellow Holds
# ----------------------------------------------------------------------------------------------------------------------

yellow_img = img.copy()
yellow_img = cv.GaussianBlur(yellow_img, (5, 5), 0)

# remove all but the brightest and most saturated pink colors
for x in yellow_img:
    for px in x:
        if px[HUE] < normal_hue(40) or px[HUE] > normal_hue(65) or px[SAT] < normal_sat(30) or px[VAL] < normal_val(45):
            px[VAL] = 0
print_img("Yellow Holds", yellow_img, NONE, True)

# apply morphological opening to remove any small artifacts that are not holds and create a defined edge for holds
avgKernel = np.ones((8, 8), np.uint8)
yellow_img = cv.morphologyEx(yellow_img, cv.MORPH_OPEN, avgKernel)
print_img("Yellow Holds Opening", yellow_img, NONE, True)

# apply morphological closing to make the holds to fill in chalky areas
avgKernel = np.ones((35, 35), np.uint8)
yellow_img = cv.morphologyEx(yellow_img, cv.MORPH_CLOSE, avgKernel)
print_img("Yellow Holds Closing", yellow_img, NONE, True)

# apply morphological opening again, this time eliminating larger artifacts
avgKernel = np.ones((15, 15), np.uint8)
yellow_img = cv.morphologyEx(yellow_img, cv.MORPH_OPEN, avgKernel)
print_img("Yellow Holds Opening", yellow_img, NONE, True)

# Make the holds solid, saturated pink for an even texture
for index_x, x in enumerate(yellow_img):
    for index_px, px in enumerate(x):
        if px[SAT] < normal_sat(90):
            px[SAT] = normal_sat(90)
        if px[VAL] > normal_val(30):
            px[HUE] = normal_hue(55)
            px[VAL] = normal_val(90)
            img_final[index_x, index_px] = px

print_img("Yellow Holds Solid", yellow_img, NONE, True)

# Identify the boundaries of each hold and mark them on the original image
yellow_gray = cv.cvtColor(yellow_img, cv.COLOR_HSV2BGR)
yellow_gray = cv.cvtColor(yellow_gray, cv.COLOR_BGR2GRAY)
contours, hierarchy = cv.findContours(yellow_gray, 1, 2)
for contour in contours:
    convexHull = cv.convexHull(contour)
    cv.drawContours(img_final, [convexHull], -1, (0, 0, 0), 7)
print_img("Yellow Holds Contour", img_final, NONE, True)
