#======================================================================================================================
# ECE 6258 - Digital Image Processing
# Semester Project - Object Recognition with Limited Data Sets and Complex Environments
# Author: Jordan Leiker
# GTid: 903453031
#
# This file performs the CURE-OR image prediction algorithm. It uses two convolutional neural nets, image processing,
# template matching and more to accomplish this. The structure of this algorithm is as follows:
#
# (1) Image Pre-process
# (2) Distortion Prediction (CNN-B)
# (3) Image Clean-up
# (4) Template Matching
# (5) Background smoothing
# (6) Object category prediction (CNN-A)
#
#======================================================================================================================


import utility_functions as util
import dippykit as dip
import numpy as np
import cv2
import copy
import cnna_objects as cnna
import cnnb_distortions as cnnb
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


DEBUG_MODE = 0
SUPPRESS_IMAGE_SHOW_UNTIL_END = 1


def cureor_predict(fpath_test_image, image_size, num_objects, cnn_model_object, cnn_model_distort, bypass_mode=0):

    M_max = image_size[0]
    N_max = image_size[1]


    # -----------------------------------------------------------------------------------------------------------------
    # (0) Predict Only Option
    #     Note: Used for debug/comparison of results
    # -----------------------------------------------------------------------------------------------------------------
    if bypass_mode == 1:
        img = util.load_image(fpath_test_image)
        img = util.resize_image(img, M_max, N_max, pad_type='reflect')  # keras expects 968x648 sized images
        prediction_object = cnna.cnna_predict(img, cnn_model_object)
        return prediction_object


    # -----------------------------------------------------------------------------------------------------------------
    # (1) Image Pre-process
    #  1.a Convert all images to RBG. The model expects 3-channel images for predictions.
    #  1.b Resample all images to 968x648; the model expects this size.
    # -----------------------------------------------------------------------------------------------------------------

    img = util.load_image(fpath_test_image)
    img = util.resample_image(img, M_max, N_max)

    if DEBUG_MODE == 1:
        print("Image shape:", img.shape)
        print("Image type:", img.dtype)
        dip.figure()
        dip.imshow(img)
        dip.title('Original Image')
        if not(SUPPRESS_IMAGE_SHOW_UNTIL_END):
            dip.show()


    # -----------------------------------------------------------------------------------------------------------------
    # (2) Distortion Prediction (CNN-B)
    #        Note: Predictions are as follows:
    #         [0] undersaturation
    #         [1] oversaturation
    #         [2] guassian blur
    #         [3] contrast
    #         [4] dirty lens white/1
    #         [5] dirty lens black/2
    #         [6] salt and pepper
    #
    #        Note2: The CNN will detect S&P and guassian distortion; it cannot detect an "absence" of noise.
    #               Because S&P and Gauss are the only distortions being corrected, this is OK. Thus, unless the
    #               distortion predicted is S&P or Gauss we do nothing.
    # -----------------------------------------------------------------------------------------------------------------

    prediction_distortion = cnnb.cnnb_predict(img, cnn_model_distort)


    # -----------------------------------------------------------------------------------------------------------------
    # (3) Image Clean-up
    #  3.a If distortion predicted is salt & pepper, perform median filtering
    #  3.b If distortion is gaussian perform laplacian sharpening
    #  3.c If distortion is under/oversaturation or low contrast, due histogram equalization ***NOT USED***
    #  3.d If anything else, do nothing.
    # -----------------------------------------------------------------------------------------------------------------

    if prediction_distortion[0] == 2:
        # Extended Laplace Kernel
        kernel = np.zeros((3, 3))
        kernel.fill(1)
        idx = int(kernel.shape[0] / 2)
        val = 1 - (kernel.shape[0] * kernel.shape[1])
        kernel[idx, idx] = val
        img_clean,_ = util.sharpen(img, kernel)
    elif prediction_distortion[0] == 6:
        img_clean = util.median_filt_bank(img)
    else:
        img_clean = img

    if bypass_mode == 2:
        prediction_object = cnna.cnna_predict(img_clean, cnn_model_object)
        return prediction_object

    if DEBUG_MODE == 1:
        print("Distortion Prediction:", prediction_distortion[0])
        dip.figure()
        dip.subplot(1,2,1)
        dip.imshow(img)
        dip.title('Original Image')
        dip.subplot(1,2,2)
        dip.imshow(img_clean)
        dip.title('Cleaned Image')
        if not(SUPPRESS_IMAGE_SHOW_UNTIL_END):
            dip.show()


    # -----------------------------------------------------------------------------------------------------------------
    # (4) Template Matching
    #  4.a Load object templates
    #  4.b Prepare image
    #  4.b.i   RGB -> Gray
    #  4.b.ii  Pad Edges
    #  4.b.iii Dither (add S&P) if not already S&P'ed
    #  4.c Template match for all objects
    #  4.d Find highest correlation index of all template matches
    # -----------------------------------------------------------------------------------------------------------------

    # 4.a Load object templates
    fname = 'template_master.jpg'
    template = dip.imread('templates/' + fname)
    template = dip.rgb2gray(template)

    if DEBUG_MODE == 1:
        dip.figure()
        dip.imshow(template, 'gray')
        dip.title('Morphed Matching Template')
        if not(SUPPRESS_IMAGE_SHOW_UNTIL_END):
            dip.show()

    # 4.b Prepare image
    img_gray = dip.rgb2gray(img)
    img_pad = util.resize_image(img_gray, M_max + int(template.shape[0]), N_max + int(template.shape[1]), 'edge')
    # Only add S&P / dithering if not already noisy
    if prediction_distortion[0] == 6:
        img_temp_match = img_pad
    else:
        img_temp_match = img_pad
        img_temp_match = dip.im_to_float(img_temp_match)
        img_temp_match = util.add_salt_and_pepper(img_temp_match, 0.25)
        img_temp_match = dip.float_to_im(img_temp_match)

    #  4.c Template match for all objects
    #  4.d Find highest correlation index of all template matches
    temp_match_result = cv2.matchTemplate(img_temp_match, template, cv2.TM_CCOEFF_NORMED)
    minVal, MaxVal, minLoc, maxLoc = cv2.minMaxLoc(temp_match_result)

    if DEBUG_MODE == 1:
        print('Best correlation index:', minLoc)
        dip.figure()
        dip.subplot(1, 3, 1)
        dip.imshow(img)
        dip.title('Original Image')
        dip.subplot(1, 3, 2)
        dip.imshow(img_temp_match, 'gray')
        dip.title('Padded and Dithered for TempMatching')
        dip.subplot(1, 3, 3)
        dip.imshow(temp_match_result)
        dip.title('Template Matching Result')
        dip.colorbar()
        if not(SUPPRESS_IMAGE_SHOW_UNTIL_END):
            dip.show()


    # -----------------------------------------------------------------------------------------------------------------
    # (5) Background smoothing
    #  5.a Create a mask to cut out as much background around the image as possible (ellipse is used here)
    #  5.b Blur the original image to reduce the effects of the 3D background
    #  5.c Set the blurred image equal to the new mask we create that preserved the object
    # -----------------------------------------------------------------------------------------------------------------

    # 5.a Create a mask
    #   Note: the ellipse mask is shrunk in by 10% to reduce 3D background more. The assumption is not enough of
    #         the object will be cut off to make a difference, but any background cutout helps
    mask = np.zeros(template.shape)
    cv2.ellipse(img=mask,
                center=(int(mask.shape[1] / 2), int(mask.shape[0] / 2)),
                axes=(int((9 / 10) * (mask.shape[1] / 2)), int((9 / 10) * (mask.shape[0] / 2))),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(1, 1, 1),
                thickness=-1)

    mask = dip.float_to_im(mask)
    mask_inv = cv2.bitwise_not(mask)

    #  5.b Blur the original image to reduce the effects of the 3D background
    # First resize the image so that we can fit the template on top of the the ROI completely
    img_resize = util.resize_image(img_clean, M_max + int(template.shape[0]), N_max + int(template.shape[1]), 'reflect')
    # Fix match location to account for resizing
    minLoc0 = minLoc[0] + int(template.shape[1]/2)
    minLoc1 = minLoc[1] + int(template.shape[0]/2)
    minLocTemp = (minLoc0, minLoc1)

    # Blur image
    window = dip.window_2d((25, 25), window_type='gaussian', var=50)
    img_blur = img_resize.astype(np.float64)
    img_blur /= 255
    img_blur = dip.convolve2d(img_blur, window)
    img_blur *= 255
    img_blur = img_blur.astype(np.uint8)

    # The ROI is the template sized section around the object
    img_blur_roi = img_blur[int(minLocTemp[1] - mask.shape[0] / 2):int(minLocTemp[1] + mask.shape[0] / 2),
                            int(minLocTemp[0] - mask.shape[1] / 2):int(minLocTemp[0] + mask.shape[1] / 2)]
    img_clean_roi = img_resize[int(minLocTemp[1] - mask.shape[0] / 2):int(minLocTemp[1] + mask.shape[0] / 2),
                               int(minLocTemp[0] - mask.shape[1] / 2):int(minLocTemp[0] + mask.shape[1] / 2)]

    if DEBUG_MODE:
        print('img_clean_roi.shape:', img_clean_roi.shape)
        print('img_blur_roi.shape:', img_blur_roi.shape)
        print('mask.shape:', mask.shape)
        print('mask_inv.shape:', mask_inv.shape)

    img_clean_masked_roi = cv2.bitwise_and(img_clean_roi, img_clean_roi, mask=mask)
    img_blur_masked_inv_roi = cv2.bitwise_and(img_blur_roi, img_blur_roi, mask=mask_inv)
    img_roi = img_clean_masked_roi + img_blur_masked_inv_roi

    img_masked = copy.deepcopy(img_blur)
    img_masked[int(minLocTemp[1]-mask.shape[0]/2):int(minLocTemp[1]+mask.shape[0]/2),
             int(minLocTemp[0]-mask.shape[1]/2):int(minLocTemp[0]+mask.shape[1]/2)]= img_roi

    # Crop to original size
    img_predict = dip.resample(img_masked, np.array([[1, 0], [0, 1]]), crop=True, crop_size=(M_max, N_max))

    if DEBUG_MODE == 1:
        print('img_predict.shape', img_predict.shape)
        dip.figure()
        dip.subplot(1, 3, 1)
        dip.imshow(img_clean)
        dip.title('Cleaned Image')
        dip.subplot(1, 3, 2)
        dip.imshow(img_masked, 'gray')
        dip.title('Masked Image')
        dip.subplot(1, 3, 3)
        dip.imshow(img_predict)
        dip.title('Image for Prediction')
        if not(SUPPRESS_IMAGE_SHOW_UNTIL_END):
            dip.show()


    # -----------------------------------------------------------------------------------------------------------------
    # (6) Object category prediction (CNN-A)
    # -----------------------------------------------------------------------------------------------------------------

    prediction_object = cnna.cnna_predict(img_predict, cnn_model_object)

    if DEBUG_MODE == 1:
        print('Object Prediction:', prediction_object)


    if SUPPRESS_IMAGE_SHOW_UNTIL_END:
        dip.show()

    return prediction_object