#----------------------------------------------------------------------------------------------------
# This file sorts the single training image file in to the directory structure needed to train both
# of the CNN's. The steps are
#
# 1) Create a folder for each object type under 'cnna/train/' and then copy the images to their
# respective folders
# 2) Create a folder for each object type under 'cnna/val/' and then move 10% of each object type
# to the new valid folder (the 10% is chosen randomly)
# 3) Repeat with 'cnnb/train/' and 'cnnb/val/', which are based on distortion type.
#    NOTE0: for cnnb, distortion types 1-9 are combined with 10-18 (as 10-18 are just grayscale)
#    NOTE1: for cnnb, distortion types 1,2,10,11 are OMITTED from training. As there is no real
#           distortions (1 and 10 are normal, 2 and 11 are resized)
#----------------------------------------------------------------------------------------------------


import numpy as np
import os
from shutil import copyfile

# Skip sorting of train data (because it's prob already done)
SKIP_TRAIN_SORT = 1
SKIP_VAL_CLASS_SORT = 0
SKIP_VAL_DISTORTION_SORT = 1

PATH_TRAIN_CSV = "train.csv"
PATH_TRAIN_DATA = "train/"
# CNN-A is category CNN
PATH_CNNA_TRAIN_DATA = "cnna/train/"
PATH_CNNA_VAL_DATA = "cnna/val/"
# CNN-B is distortion CNN
PATH_CNNB_TRAIN_DATA = "cnnb/train/"
PATH_CNNB_VAL_DATA = "cnnb/val/"


#----------------------------------------------------------------------------------------------------
# Read in CSV file containing ground truth on training data
#----------------------------------------------------------------------------------------------------
# train.csv is in format:
# (0) imageID (1) class (2) background (3) perspective (4) challengeType (5) challengeLevel
train_metadata = np.genfromtxt(PATH_TRAIN_CSV, delimiter=',')
# Note: Skip first row (column labels)
y_train_class = train_metadata[1:, 1]
y_train_distortion_type = train_metadata[1:, 4]
y_train_distortion_level = train_metadata[1:, 5]
num_samples = y_train_class.shape[0]


#----------------------------------------------------------------------------------------------------
# First sort all of the training images (images are copied, not moved)
#----------------------------------------------------------------------------------------------------
if not(SKIP_TRAIN_SORT):
    # Sort by category
    print("Sorting by category")
    for i in range(num_samples):
        fname = str(i).zfill(5) + ".jpg"
        ifpath = PATH_TRAIN_DATA
        im_class = y_train_class[i]
        ofpath = PATH_CNNA_TRAIN_DATA + str(int(im_class)) + '/'

        if not os.path.exists(ofpath):
            os.makedirs(ofpath)

        copyfile(ifpath+fname, ofpath+fname)

    # Sort by distortion
    print("Sorting by distortion")
    for i in range(num_samples):
        fname = str(i).zfill(5) + ".jpg"
        ifpath = PATH_TRAIN_DATA
        im_distort_type = y_train_distortion_type[i]

        # Do not sort 1,2,10,11 as we don't want to train on them
        if im_distort_type == 1 or im_distort_type == 2 or im_distort_type == 10 or im_distort_type == 11:
            continue

        # For 10-18, write the files to 1-9 (because distortions are the same, just grayscale)
        if im_distort_type > 9:
            im_distort_type = im_distort_type - 9
        else:
            im_distort_type = im_distort_type

        ofpath = PATH_CNNB_TRAIN_DATA + str(int(im_distort_type)) + '/'

        if not os.path.exists(ofpath):
            os.makedirs(ofpath)

        copyfile(ifpath+fname, ofpath+fname)


#----------------------------------------------------------------------------------------------------
# Move off 10% of the training data to a validation directory
#----------------------------------------------------------------------------------------------------

TOTAL_NUM_FILES = num_samples
PERCENTAGE_TO_WITHOLD_FOR_VALID = 10


# Create validation data for category
if not(SKIP_VAL_CLASS_SORT):

    TOTAL_NUM_CLASSES = 10
    NUM_FILES_PER_CLASS = TOTAL_NUM_FILES/TOTAL_NUM_CLASSES
    NUM_VALIDATION_FILES = NUM_FILES_PER_CLASS*(PERCENTAGE_TO_WITHOLD_FOR_VALID/100)

    print("Creating Category Validation Set")
    for i in range(TOTAL_NUM_CLASSES):
        print("Category:", i+1)
        for j in range(int(NUM_VALIDATION_FILES)):
            rand_file_idx = np.random.randint(0, TOTAL_NUM_FILES)
            fname = str(rand_file_idx).zfill(5) + ".jpg"
            train_path = (PATH_CNNA_TRAIN_DATA + str(i+1) + '/' + fname)
            exists = os.path.isfile(train_path)

            # Check if source file exists, if not, keep trying
            while(not exists):
                rand_file_idx = np.random.randint(0, TOTAL_NUM_FILES)
                fname = str(rand_file_idx).zfill(5) + ".jpg"
                train_path = (PATH_CNNA_TRAIN_DATA + str(i+1) + '/' + fname)
                exists = os.path.isfile(train_path)

            # Check if the destination path exists, if not make it
            exists = os.path.isdir(PATH_CNNA_VAL_DATA + str(i+1))
            if not(exists):
                os.makedirs(PATH_CNNA_VAL_DATA + str(i+1))

            os.rename(train_path, (PATH_CNNA_VAL_DATA + str(i+1) + '/' + fname))



# Create validation data for distortions
# Note: 1,2,10,11 ommitted as we don't want to train on "no distortion" distortions
if not(SKIP_VAL_DISTORTION_SORT):

    DISTORTIONS_TO_SORT = [3,4,5,6,7,8,9,12,13,14,15,16,17,18]
    TOTAL_NUM_DISTORTIONS = 18
    NUM_FILES_PER_DISTORTION = TOTAL_NUM_FILES/TOTAL_NUM_DISTORTIONS
    NUM_VALIDATION_FILES = NUM_FILES_PER_DISTORTION*(PERCENTAGE_TO_WITHOLD_FOR_VALID/100)

    print("Creating Distortion Validation Set")
    for i in DISTORTIONS_TO_SORT:
        print("Distortion:", i)
        for j in range(int(NUM_VALIDATION_FILES)):
            rand_file_idx = np.random.randint(0, TOTAL_NUM_FILES)
            fname = str(rand_file_idx).zfill(5) + ".jpg"
            train_path = (PATH_CNNB_TRAIN_DATA + str(i) + '/' + fname)
            exists = os.path.isfile(train_path)

            # Check if source file exists, if not, keep trying
            while(not exists):
                rand_file_idx = np.random.randint(0, TOTAL_NUM_FILES)
                fname = str(rand_file_idx).zfill(5) + ".jpg"
                train_path = (PATH_CNNB_TRAIN_DATA + str(i) + '/' + fname)
                exists = os.path.isfile(train_path)

            # For 10-18, write the files to 1-9 (because distortions are the same, just grayscale)
            if i > 9:
                idx = i - 9
            else:
                idx = i

            # Check if the destination path exists, if not make it
            exists = os.path.isdir(PATH_CNNB_VAL_DATA + str(idx))
            if not(exists):
                os.makedirs(PATH_CNNB_VAL_DATA + str(idx))

            os.rename(train_path, (PATH_CNNB_VAL_DATA + str(idx) + '/' + fname))
