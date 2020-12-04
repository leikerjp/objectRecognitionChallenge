#======================================================================================================================
# ECE 6258 - Digital Image Processing
# Semester Project - Object Recognition with Limited Data Sets and Complex Environments
# Author: Jordan Leiker
# GTid: 903453031
#
# This is a script to create the morphed object matching templates.
#======================================================================================================================

import numpy as np
import dippykit as dip
import cv2
import utility_functions as util

PATH_TO_BASIC_IMAGES = 'template_crops/'
PATH_OUT_TEMPLATE_IMAGES = 'templates/'
M_max = 968
N_max = 648


# Templates are based on first 50 images with white background.
#   Note: Manually cropped images were used and thresholds tweaked by hand as needed.
idx_stride = 10*np.arange(5)

# Image numbers, e.g. frying pan is images 3, 13, 23, 33, 43
camera = 0 + idx_stride
traffic_cone = 1 + idx_stride
baseball = 2 + idx_stride
frying_pan = 3 + idx_stride
giraffe = 4 + idx_stride
cell_phone = 5 + idx_stride
hair_brush = 6 + idx_stride
label_maker = 7 + idx_stride
vitamin = 8 + idx_stride
shoe = 9 + idx_stride

objects = [camera,
           traffic_cone,
           baseball,
           frying_pan,
           giraffe,
           cell_phone,
           hair_brush,
           label_maker,
           vitamin,
           shoe]

accum_total = np.zeros((M_max,N_max,3), dtype=np.uint8)
mask = np.zeros((M_max,N_max,3), dtype=np.uint8)
template_id = 1

for idx_object in objects:
    # Reset object accumulator each object
    accum_object = np.zeros((M_max, N_max, 3), dtype=np.uint8)
    for idx_image in range(len(idx_stride)):

        # (1) Load and convert to gray
        img = util.load_image(PATH_TO_BASIC_IMAGES + str(idx_object[idx_image]).zfill(5) + ".jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # (2) Threshold
        th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # (3) Find the min-area contour
        _, cnts, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(cnts, key=cv2.contourArea)
        cnt = cntsSorted[-1]

        # (4) Create mask and do bitwise-op
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        dst = cv2.bitwise_and(img, img, mask=mask)

        # (5) Crop image to JUST around the object, resize so object is in center
        x, y, w, h = util.find_bounding_rect(dst)
        dstcrop = dst[y:y + h, x:x + w]
        dstcrop = util.resize_image(dstcrop, M_max, N_max, 'constant')

        # (6) Logical OR - for accumulation of images
        accum_object = cv2.bitwise_or(dstcrop, accum_object)
        accum_total = cv2.bitwise_or(dstcrop, accum_total)

    x, y, w, h = util.find_bounding_rect(accum_object)
    accum_object = accum_object[y:y + h, x:x + w]
    dip.im_write(accum_object, PATH_OUT_TEMPLATE_IMAGES + 'template_' + str(template_id).zfill(2) + ".jpg")
    template_id = template_id + 1

x, y, w, h = util.find_bounding_rect(accum_total)
accum_total = accum_total[y:y + h, x:x + w]
dip.im_write(accum_total, PATH_OUT_TEMPLATE_IMAGES + 'template_master.jpg')


