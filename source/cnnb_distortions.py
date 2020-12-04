# -----------------------------------------------------------------------------------------------------------------
# Distortion Classification CNN - Functions for loading from a set of weights and predicting
# -----------------------------------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np


def load_model_cnnb(weights=None, verbose=0):

    model_b = Sequential()
    model_b.add(Conv2D(32, (3, 3), padding='same', input_shape=(968, 648, 3)))
    model_b.add(Activation('relu'))
    model_b.add(MaxPooling2D(pool_size=(2, 2)))

    model_b.add(Conv2D(32, (3, 3)))
    model_b.add(Activation('relu'))
    model_b.add(MaxPooling2D(pool_size=(2, 2)))

    model_b.add(Conv2D(64, (3, 3)))
    model_b.add(Activation('relu'))
    model_b.add(MaxPooling2D(pool_size=(2, 2)))

    model_b.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model_b.add(Dense(128))
    model_b.add(Activation('relu'))
    model_b.add(Dropout(0.5))
    model_b.add(Dense(7))
    model_b.add(Activation('softmax'))

    if verbose:
        model_b.summary()

    model_b.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    if weights:
        print("Loading Weights:", weights)
        model_b.load_weights(weights)
    else:
        print('No weights to load, starting model from scratch!')

    return model_b


def cnnb_predict(img, model):

    # keras expects an array of predictions
    img_pred = img[np.newaxis, :]
    prediction = model.predict_classes(img_pred, verbose=0)
    return prediction
