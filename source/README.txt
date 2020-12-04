=======================================================================================================================
(1) Setup
=======================================================================================================================
The following are the required tools and libraries to run the included scripts:
Note: Automatically installed libraries are not included below, e.g. NumPy for Dippykit

Python 3.6
Dippykit		
Keras			
TensorFlow		


If starting from scratch within a virtual env (using Python 3.6) such as PyCharm:

From the terminal:
1) pip install dippykit
2) pip install keras
3) pip install tensorflow

Manual steps:
1) Extract all of the files in LeikerJordan.zip
2) Copy and paste all extracted files and folders in to work environment.


=======================================================================================================================
(2) Running the Scripts
=======================================================================================================================
To run the scripts from the command line:
python run_me.py NUM_IMAGES_TO_CLASSIFY (default is all images in "test/" folder, max of 6600 - starting at 09900.jpg)

examples:
python run_me.py 	   (will attempt to classify each image in the "test/" folder local to the project)
python run_me.py 5     (will attempt to classify the first 5 images in the "test/" folder local to the project)

- Note: An abridged test set is included to accelerate the script testing process. It contains 10 arbitrarily chosen
images from the real test set. Should the user wish to point it at a different test set see the 'advanced' section below
- Note2: It is not recommended to try to classify more than the 10 images..for sanities sake. It takes forever... (about
10 seconds per prediction)

Advanced:
If the user would like to point the scripts at their own test image directory, the following line in 'run_me.py' must
be edited:
'''
#----------------------------------------------------------------------------------------------------------------------
# If the provided test images are not used, please set a new path here.
#----------------------------------------------------------------------------------------------------------------------
PATH_TO_TEST_SET = 'test/'
'''


=======================================================================================================================
(3) Outputs
=======================================================================================================================
Predictions are printed to the terminal window, additionally, a CSV file containing two columns composed of the image
index and prediction number is saved to the local project directory.




=======================================================================================================================
(Appendix) Description of included files and folders
=======================================================================================================================
#################################### Python Files ####################################
run_me.py -
The main script that generates a file path, calls the algorithm, and writes all outputs to a csv file

image_predict.py -
The algorithm. Loads images, performs predictions and all image processing.

utility_functions.py -
Useful functions that are used for image processing.

cnna_objects.py -
The object classification CNN model.

cnnb_distortions.py -
The distortion classification CNN model.

create_templates.py -
The script used to create the 'morph' template and object  templates

sort_images.py -
The script used to sort all of the images by class or distortion type for the model training.

cnna_objects_train.py -
This file is what's used to train CNNB
Note: Provided for completeness. NOT intended to be run.

cnnb_distortions_train.py -
This file is what's used to train CNNB
Note: Provided for completeness. NOT intended to be run.

#################################### Folders and Misc ####################################
templates/ -
Required for predictions. This folder contains all the object templates and the 'morph' template that's used

test/ -
20 arbitrarily chosen test images (out of the total 6600 test images)

orcure_weights_4_07.h5 -
Weights for the trained CNNA (object classifcation). 7 epochs, ~75% acc, 80% val-acc)

orcure_dist_weights_1_05 -
Weights for the trained CNNB (distortion classifcation). 5 epochs, ~98% acc, 99% val-acc)

template_crops/ -
The starting point for the template creation