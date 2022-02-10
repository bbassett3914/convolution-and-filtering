"""

@author: Brendan Bassett
@Date: 10/27/2021

CS3150 Image Processing
Dr. Feng Jiang
Metropolitan State University of Denver

An exercise in face detection using the RGB simple skin classifying method.
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


BMP = ".bmp"
FACE_GOOD = "face_good"
FACE_DARK = "face_dark"

OUTPUT_LOCATION = "C:\\Users\\bbass\\PycharmProjects\\Skin_Detection\\output_images\\"

NONE = 0
OPENCV = 1
PYPLOT = 2


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

# Helper for saving image files and displaying them
def print_img(title: str, img, show_type: int = PYPLOT, save_image=False):
    if save_image:
        bgrImg = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(OUTPUT_LOCATION + title + BMP, bgrImg)

    if show_type is OPENCV:
        cv.imshow(title, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif show_type is PYPLOT:
        plt.figure()
        plt.title(title)
        plt.imshow(img)
        plt.waitforbuttonpress()


# Applies gamma correction to an rgb image
def gamma_correction_rgb(img, gamma):
    luv_img = cv.cvtColor(img, cv.COLOR_RGB2LUV)

    # extract luminance channel
    l = luv_img[:, :, 0]
    # normalize
    l = l / 255.0
    # apply power transform
    l = np.power(l, gamma)
    # scale back
    l = l * 255

    luv_img[:, :, 0] = l.astype(np.uint8)
    img = cv.cvtColor(luv_img, cv.COLOR_LUV2RGB)

    return img


# Applies skin detection via thresholding to an rgb image
def skin_threshold_rgb(img):
    # extract color channels and save as SIGNED ints
    # need the extra width to do subtraction
    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)

    skin_mask = (r > 95) & (g > 40) & (b > 20) & ((img.max() - img.min()) > 15) & (np.abs(r - g) > 15) & (r > g) \
                & (r > b)

    return img * skin_mask.reshape(skin_mask.shape[0], skin_mask.shape[1], 1)


# Finds the minimum between two large peaks in a given histogram
def find_local_min(histogram):
    # a local-minimum kernel
    kern = np.array(
        [2, 0, 0, 0,
         2, 0, 0, 0,
         2, 0, 0, 0,
         2, 0, 0, 0,
         1, 0, 0, 0,
         1, 0, 0, 0,
         1, 0, 0, 0,
         1, 0, 0, 0,
         -3, -3, -3, -3
         - 3, -3, -3, -3
            , 0, 0, 0, 1
            , 0, 0, 0, 1
            , 0, 0, 0, 1
            , 0, 0, 0, 1
            , 0, 0, 0, 2
            , 0, 0, 0, 2
            , 0, 0, 0, 2
            , 0, 0, 0, 2])

    # remove the large "spike" representing all the completely dark pixels
    histogram[0] = 0

    # use the kernel to find the local minimum (where the convolution produced the greatest result)
    deriv = np.convolve(histogram, kern, mode='same')
    local_min = deriv.argmax()

    return local_min


# ======================================================================================================================
# MAIN CODE
# ======================================================================================================================

# load and display sample image
img_dark = cv.imread(FACE_DARK + BMP)
img_dark = cv.cvtColor(img_dark, cv.COLOR_BGR2RGB)
print_img(FACE_DARK, img_dark, NONE)

# make the dark image lighter
img_dark = gamma_correction_rgb(img_dark, 0.70)
print_img(FACE_DARK + "_gamma", img_dark, NONE)

# apply the skin detection
skin_threshold_dark = skin_threshold_rgb(img_dark)
print_img(FACE_DARK + "_skin_detected", skin_threshold_dark, NONE)

# calculate the histogram
img_dark_luv = cv.cvtColor(img_dark, cv.COLOR_RGB2LUV)
hist = cv.calcHist([img_dark_luv], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.show()

# convert the histogram to a 1-dimensional array
hist = hist.flatten()

# find the threshold
threshold = find_local_min(hist)

# set the luminosity to zero for every pixel with luminosity that exceeds the threshold
for x in img_dark_luv:
    for px in x:
        if px[0] > threshold:
            px[0] = 0

img_dark = cv.cvtColor(img_dark_luv, cv.COLOR_LUV2RGB)
print_img(FACE_DARK + "_histogram_threshold", img_dark, NONE)

# detect the skin again
img_dark_skin_threshold = skin_threshold_rgb(img_dark)
print_img(FACE_DARK + "_skin_detected_after_histogram", img_dark_skin_threshold)

# apply morphological opening to remove any artifacts outside the face
avgKernel = np.ones((6, 6), np.uint8)
img_open = cv.morphologyEx(img_dark_skin_threshold, cv.MORPH_OPEN, avgKernel)
print_img(FACE_DARK + "_skin_open", img_open, PYPLOT, True)

# apply morphological closing to remove any artifacts within the face
avgKernel = np.ones((20, 20), np.uint8)
img_close = cv.morphologyEx(img_open, cv.MORPH_CLOSE, avgKernel)
print_img(FACE_DARK + "_skin_close", img_close, PYPLOT, True)

# apply morphological opening with a larger kernel to remove the ears and chest, leaving the face and neck
avgKernel = np.ones((37, 37), np.uint8)
img_open_again = cv.morphologyEx(img_close, cv.MORPH_OPEN, avgKernel)
print_img(FACE_DARK + "_skin_open_again", img_open_again, PYPLOT, True)

# create an image which shows only the face and neck region of the ORIGINAL image
img_dark_final_luv = cv.cvtColor(img_open_again, cv.COLOR_RGB2LUV)
for index_x, img_x in enumerate(img_dark_final_luv):
    for index_y, px in enumerate(img_x):
        if px[0] == 0:
            img_dark_luv[index_x, index_y, 0] = 0

img_dark_cropped = cv.cvtColor(img_dark_luv, cv.COLOR_LUV2RGB)
print_img(FACE_DARK + "_skin_dark_cropped", img_dark_cropped, PYPLOT, True)

# identify the exterior boundary of the face
convex_image = img_dark_cropped
contours, hierarchy = cv.findContours(cv.cvtColor(img_dark_cropped, cv.COLOR_BGR2GRAY), 1, 2)

for contour in contours:
    convexHull = cv.convexHull(contour)
    cv.drawContours(img_dark_cropped, [convexHull], -1, (255, 0, 0), 2)

print_img(FACE_DARK + "_convex_hull", img_dark_cropped, PYPLOT, True)
