# -----------------------------------------------------------------------------------------------------------------
# Object Classification CNN - Functions for loading from a set of weights and predicting
# -----------------------------------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np


def load_model_cnna(weights=None, verbose=0):
    model_a = Sequential()

    model_a.add(Conv2D(32, (3, 3), padding='same', input_shape=(968, 648, 3)))
    model_a.add(Activation('relu'))
    model_a.add(MaxPooling2D(pool_size=(2, 2)))

    model_a.add(Conv2D(32, (3, 3)))
    model_a.add(Activation('relu'))
    model_a.add(MaxPooling2D(pool_size=(2, 2)))

    model_a.add(Conv2D(64, (3, 3)))
    model_a.add(Activation('relu'))
    model_a.add(MaxPooling2D(pool_size=(2, 2)))

    model_a.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model_a.add(Dense(128))
    model_a.add(Activation('relu'))
    model_a.add(Dropout(0.5))
    model_a.add(Dense(10))
    model_a.add(Activation('softmax'))

    if verbose:
        model_a.summary()

    model_a.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    if weights:
        print("Loading Weights:", weights)
        model_a.load_weights(weights)
    else:
        print('No weights to load, starting model from scratch!')

    return model_a


def cnna_predict(img, model):

    # keras expects an array of predictions
    img_pred = img[np.newaxis, :]
    prediction = model.predict_classes(img_pred, verbose=0)
    return prediction