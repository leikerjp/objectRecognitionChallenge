# -----------------------------------------------------------------------------------------------------------------
# Object Classification CNN for Training - This is the file used to train the CNNA.
# Note: This is included for completeness. It is NOT part of the algorithm scripts and is NOT intended to be run
# -----------------------------------------------------------------------------------------------------------------



from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

ROOT_DIR = "C:/Users/Jordan/Documents/GeorgiaTech/ECE6258_DigitalImageProcessing/Project/backup_project_images/"
PATH_TRAIN_DATA = ROOT_DIR + "train/sort_train/"
PATH_VAL_DATA = ROOT_DIR + "train/sort_val/"


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(968, 648, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect'
        )

val_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode='reflect'
        )

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        PATH_TRAIN_DATA,  # this is the target directory
        target_size=(968, 648),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = val_datagen.flow_from_directory(
        PATH_VAL_DATA,
        target_size=(968, 648),
        batch_size=batch_size,
        class_mode='categorical')

# Load weights if continuing to train from saved weights
#model.load_weights('orcure_weights_4_07.h5')

model.fit_generator(
        train_generator,
        steps_per_epoch=8910 // batch_size,  # Total num samples / batch_size (recommended by keras)
        epochs=1,  # 50
        validation_data=validation_generator,
        validation_steps=990 // batch_size)

model.save_weights('orcure_weights_5_01.h5')  # always save your weights after training or during training

print("Duuuuuuuuuuude we made it!")
