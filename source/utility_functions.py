#======================================================================================================================
# ECE 6258 - Digital Image Processing
# Semester Project - Object Recognition with Limited Data Sets and Complex Environments
# Author: Jordan Leiker
# GTid: 903453031
#
# These are some useful functions
#======================================================================================================================



import dippykit as dip
import numpy as np
import cv2


def load_image(fpath) -> np.ndarray:

    im = dip.im_read(fpath)

    #im = dip.im_to_float(im)
    img_rgb = np.empty((im.shape[0], im.shape[1], 3), dtype='uint8')

    if im.ndim == 2:
        # Conv grayscale to RGB (quick approach)
        img_rgb = np.dstack([im]*3)
    else:
        img_rgb[:, :, :] = im

    return img_rgb


def resize_image(im: np.ndarray, M_max, N_max, pad_type) -> np.ndarray:
    M_quo, M_rem = divmod(M_max - im.shape[0], 2)
    N_quo, N_rem = divmod(N_max - im.shape[1], 2)

    img = np.zeros((M_max, N_max, 3))

    # Handle non-grayscale cases
    if im.ndim == 3:
        im0 = im[:, :, 0]
        im1 = im[:, :, 1]
        im2 = im[:, :, 2]
        # Pad using symmetric reflection. Extra pixel width always goes on bottom and right for uneven pads
        im0 = np.pad(im0, [[M_quo, M_quo + M_rem], [N_quo, N_quo + N_rem]], pad_type)
        im1 = np.pad(im1, [[M_quo, M_quo + M_rem], [N_quo, N_quo + N_rem]], pad_type)
        im2 = np.pad(im2, [[M_quo, M_quo + M_rem], [N_quo, N_quo + N_rem]], pad_type)
        img[:, :, 0] = im0
        img[:, :, 1] = im1
        img[:, :, 2] = im2
    else:
        img = np.pad(im, [[M_quo, M_quo + M_rem], [N_quo, N_quo + N_rem]], pad_type)

    img = img.astype(np.uint8)

    return img


def resample_image(img:np.ndarray, M_max, N_max) -> np.ndarray:

    # Enlarge as close as possible to max img size
    row_scale = 1 / (M_max / img.shape[0])
    column_scale = 1 / (N_max / img.shape[1])
    M = np.array([[row_scale, 0], [0, column_scale]])  # scaling matrix
    im_intrp = dip.resample(img, M, interp='bicubic')

    # Crop / Pad (with reflect) the remaining pixels
    if im_intrp.shape[0] > M_max and im_intrp.shape[1] > N_max:
        im_intrp = dip.resample(im_intrp, np.array([[1, 0], [0, 1]]), crop=True, crop_size=(M_max, N_max))
    elif im_intrp.shape[0] > M_max and im_intrp.shape[1] < N_max:
        im_intrp = dip.resample(im_intrp, np.array([[1, 0], [0, 1]]), crop=True, crop_size=(M_max, im_intrp.shape[1]))
    elif im_intrp.shape[0] < M_max and im_intrp.shape[1] > N_max:
        im_intrp = dip.resample(im_intrp, np.array([[1, 0], [0, 1]]), crop=True, crop_size=(im_intrp.shape[0], N_max))

    im_resize = resize_image(im_intrp, M_max, N_max, 'reflect')

    return im_resize


def sharpen(im, kernel):

    if im.dtype != np.uint8:
        print('Fix data type to uint8')
        return

    #Convert to float and scale to 255
    im = dip.im_to_float(im)
    im *= 255
    im_lap = im

    im_lap = dip.convolve2d(im_lap, kernel)
    # Shift up to remove negative values
    im_lap = im_lap - np.min(im_lap)
    # Rescale to range of integers
    im_lap = 255 * (im_lap / np.max(im_lap))

    #crop to original size
    im_lap = dip.resample(im_lap, np.array([[1, 0], [0, 1]]), crop=True, crop_size=(im.shape[0], im.shape[1]))

    # Add laplacian back in, normalize, convert to uint8
    im_n = im + im_lap
    im_n = im_n - np.min(im_n)
    im_n = 255 * im_n / np.max(im_n)
    im_n = im_n.astype(np.uint8)
    im_lap = im_lap.astype(np.uint8)

    return im_n, im_lap


def median_filt_bank(img:np.ndarray) -> np.ndarray:

    # Note: Could be tuned to get better results

    im_n = img
    for i in range(2):
        im_n = cv2.medianBlur(im_n, 3)
    for i in range(2):
        im_n = cv2.medianBlur(im_n, 5)
    for i in range(3):
        im_n = cv2.medianBlur(im_n, 7)

    return im_n


def add_salt_and_pepper(gb, prob):
    '''Adds "Salt & Pepper" noise to an image.
    gb: should be one-channel image with pixels in [0, 1] range
    prob: probability (threshold) that controls level of noise'''

    rnd = np.random.rand(gb.shape[0], gb.shape[1])
    noisy = gb.copy()
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 1
    return noisy


def find_bounding_rect(imrgb: np.ndarray):

    im = dip.rgb2gray(imrgb)

    #find bounding region from one side
    for m in range(im.shape[0]):
        for n in range(im.shape[1]):
            if im[m,n] > 0:
                #print(m, n)
                y = m - 1 # substract row becase we went 'one to far'
                break
        if im[m, n] > 0:
            break

    # find bounding region from other side, get height
    for m in range(im.shape[0]-1, 0, -1):
        for n in range(im.shape[1]):
            if im[m, n] > 0:
                #print(m,n)
                h = (m - y) + 1 # add row because we went 'one to far'
                #h = m
                break
        if im[m, n] > 0:
            break

    #find bounding region from one side
    for n in range(im.shape[1]):
        for m in range(im.shape[0]):
            if im[m,n] > 0:
                #print(m, n)
                x = n- 1 # substract row becase we went 'one to far'
                break
        if im[m, n] > 0:
            break

    # find bounding region from other side, get width
    for n in range(im.shape[1]-1, 0, -1):
        for m in range(im.shape[0]):
            if im[m, n] > 0:
                #print(m, n)
                w = (n - x) + 1 # add row because we went 'one to far'
                #w = n
                break
        if im[m, n] > 0:
            break

    return x,y,w,h


def quantize(im:np.ndarray, thresholds:np.ndarray) -> np.ndarray:
    '''
    Example threshold array: thresh = np.array([0,20,30,40,50,128,255])
    creates thresholds at 10, 25, 35, 45, 84, 192
    '''
    len = thresholds.size
    im_quant = np.zeros((im.shape))
    if not(len % 2): # check if even
        print("Odd number of thresholds required")
        return
    if im.ndim > 2:
        print("Single Channel Image expected")
        return

    quant_values = np.zeros(len-1)
    for i in range(len-1):
        quant_values[i] = (thresholds[i] + thresholds[i+1])/2

    for m in range(im.shape[0]):
        for n in range(im.shape[1]):
            for i in range(len-1):
                if im[m,n] >= thresholds[i] and im[m,n] < thresholds[i+1]:
                    im_quant[m,n] = quant_values[i]

    im_quant = im_quant.astype(np.uint8)
    return im_quant

