#======================================================================================================================
# ECE 6258 - Digital Image Processing
# Semester Project - Object Recognition with Limited Data Sets and Complex Environments
# Author: Jordan Leiker
# GTid: 903453031
#
# This is the main script of the semester project targeting the CURE-OR data set.
#======================================================================================================================

import os
import image_predict as ip
import numpy as np
import datetime
import csv
import cnna_objects as cnna
import cnnb_distortions as cnnb
import sys


#----------------------------------------------------------------------------------------------------------------------
# If the provided test images are not used, please set a new path here.
#----------------------------------------------------------------------------------------------------------------------
PATH_TO_TEST_SET = 'test/'
PATH_TO_VAL_SET = 'test_val/'
PATH_PREDICTIONS = 'predictions_' #save locally

# Bypass Modes for testing / comparison
# 0 - No bypass
# 1 - Object prediction only
# 2 - Image cleanup and object prediction only
BYPASS_MODE = 0
RUN_ON_VALIDATION_SET = 0
PRINT_PREDICTIONS = 1

CNNA_WEIGHTS = 'orcure_weights_4_07.h5'
CNNB_WEIGHTS = 'orcure_dist_weights_1_05.h5'

NUM_TEST_IMAGES = 6600
STARTING_NUM_TEST_IMAGE = 9900
TOTAL_NUM_TRAIN_VAL_IMAGES = 9900
IMAGE_SIZE = (968, 648)
NUM_OBJECT_CATEGORIES = 10


if __name__ == '__main__':


    # Command Line Selection or Default
    if len(sys.argv) > 1:
        NUM_IMAGES_FOR_PREDICTION = int(sys.argv[1])
    else:
        NUM_IMAGES_FOR_PREDICTION = NUM_TEST_IMAGES


    print('Beginning CURE-OR predictions')

    # Load CNN models
    model_a = cnna.load_model_cnna(weights=CNNA_WEIGHTS, verbose=0)
    model_b = cnnb.load_model_cnnb(weights=CNNB_WEIGHTS, verbose=0)

    # -----------------------------------------------------------------------------------------------------------------
    # Run algorithm through validation data, debug/analysis only
    #   Note: This only writes MISSES to the output CSV
    # -----------------------------------------------------------------------------------------------------------------
    if RUN_ON_VALIDATION_SET:
        if not os.path.exists(PATH_TO_VAL_SET):
            print('Invalid file path: \n', PATH_TO_VAL_SET)
        else:
            print('Running on Validation Set')
            # Load truth for validation set
            train_metadata = np.genfromtxt('train.csv', delimiter=',')
            train_metadata = train_metadata[1:, :]
            predictions_made = 0
            prediction_array = np.zeros((TOTAL_NUM_TRAIN_VAL_IMAGES, 1))
            truth_array = np.zeros((TOTAL_NUM_TRAIN_VAL_IMAGES, 6))
            # Predict on all images in validation set
            for i in range(TOTAL_NUM_TRAIN_VAL_IMAGES):
                # Validation data is random, try all possible image numbers
                # until we test all of the validation images
                fname = str(i).zfill(5) + ".jpg"
                ifpath = PATH_TO_VAL_SET + fname
                exists = os.path.isfile(ifpath)
                if not exists:
                    continue
                print(i)
                # Cure-Or Algorithm
                prediction_array[predictions_made] = ip.cureor_predict(fpath_test_image=ifpath,
                                                                       image_size=IMAGE_SIZE,
                                                                       num_objects=NUM_OBJECT_CATEGORIES,
                                                                       cnn_model_object=model_a,
                                                                       cnn_model_distort=model_b,
                                                                       bypass_mode=BYPASS_MODE)
                truth_array[predictions_made] = train_metadata[i, :]

                # Correct label 0 and 1 to 1 and 10 (Keras bug)
                if prediction_array[predictions_made] == 0:
                    prediction_array[predictions_made] = 1
                elif prediction_array[predictions_made] == 1:
                    prediction_array[predictions_made] = 10

                predictions_made += 1


            # Compare results and create output array
            num_wrong = 0
            outdata = np.zeros((predictions_made, truth_array.shape[1] + 1))
            for i in range(predictions_made):
                if prediction_array[i] != truth_array[i, 1]:
                    outdata[num_wrong, 0] = prediction_array[i]
                    outdata[num_wrong, 1:] = truth_array[i]
                    num_wrong += 1

            # Create output file and write predictions in one column, correct values in other
            datecode = datetime.datetime.now().strftime('%Y%m%d%H%M')
            ofpath = PATH_PREDICTIONS + datecode + ".csv"

            with open(ofpath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                for i in range(num_wrong):
                    writer.writerow(outdata[i])

    # -----------------------------------------------------------------------------------------------------------------
    # Run algorithm through test data
    #   Note: This prints the csv file in Kaggle competition format
    # -----------------------------------------------------------------------------------------------------------------
    else:
        if not os.path.exists(PATH_TO_TEST_SET):
            print('Invalid file path: \n', PATH_TO_TEST_SET)
        else:
            print('Running on Test Set, {} Images to Test' .format(NUM_IMAGES_FOR_PREDICTION))
            predictions_made = 0
            prediction_array = np.zeros((NUM_IMAGES_FOR_PREDICTION, 1))
            prediction_idx_array = np.zeros((NUM_IMAGES_FOR_PREDICTION, 1))
            # Cycle over all test images
            for i in range(NUM_TEST_IMAGES):
                # Exit from testing if we've tested enough
                if predictions_made == NUM_IMAGES_FOR_PREDICTION:
                    break

                fname = str(i+STARTING_NUM_TEST_IMAGE).zfill(5) + ".jpg"
                ifpath = PATH_TO_TEST_SET + fname
                # Check if file exists, if not, check next
                exists = os.path.isfile(ifpath)
                if not exists:
                    continue

                print('Index:', i + STARTING_NUM_TEST_IMAGE)
                prediction_array[predictions_made] = ip.cureor_predict(fpath_test_image=ifpath,
                                                                       image_size=IMAGE_SIZE,
                                                                       num_objects=NUM_OBJECT_CATEGORIES,
                                                                       cnn_model_object=model_a,
                                                                       cnn_model_distort=model_b,
                                                                       bypass_mode=BYPASS_MODE)
                prediction_idx_array[predictions_made] = i

                # Correct label 0 and 1 to 1 and 10 (Keras bug)
                if prediction_array[predictions_made] == 0:
                    prediction_array[predictions_made] = 1
                elif prediction_array[predictions_made] == 1:
                    prediction_array[predictions_made] = 10


                if PRINT_PREDICTIONS == 1:
                    if prediction_array[predictions_made] == 1:
                        print('Camera')
                    elif prediction_array[predictions_made] == 2:
                        print('Traffic Cone')
                    elif prediction_array[predictions_made] == 3:
                        print('Baseball')
                    elif prediction_array[predictions_made] == 4:
                        print('Frying Pan')
                    elif prediction_array[predictions_made] == 5:
                        print('Giraffe')
                    elif prediction_array[predictions_made] == 6:
                        print('Cell Phone')
                    elif prediction_array[predictions_made] == 7:
                        print('Hair Brush')
                    elif prediction_array[predictions_made] == 8:
                        print('Label Maker')
                    elif prediction_array[predictions_made] == 9:
                        print('Vitamin Bottle')
                    elif prediction_array[predictions_made] == 10:
                        print('Shoe')
                    else:
                        print('Something has gone seriously wrong!')

                predictions_made += 1


            # Create output file and write predictions in one column, correct values in other
            submission_header = np.array(['imageID', 'class'])
            datecode = datetime.datetime.now().strftime('%Y%m%d%H%M')
            ofpath = PATH_PREDICTIONS + datecode + ".csv"
            submission_array = np.zeros((NUM_IMAGES_FOR_PREDICTION, 2), np.int)
            submission_array[:, 0] = STARTING_NUM_TEST_IMAGE + prediction_idx_array[:,0]
            submission_array[:, 1] = prediction_array[:, 0]

            with open(ofpath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(submission_header)
                for i in range(predictions_made):
                    writer.writerow(submission_array[i])

            print(' {} predictions requested, {} predictions completed. Thanks for your business. Please come again'
                  .format(NUM_IMAGES_FOR_PREDICTION ,predictions_made))




