from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
import numpy as np
from keras.applications import imagenet_utils
import json
from keras.applications.imagenet_utils import *
import pandas as pd
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.models import Model

train_data_dir = "/ebs/categories/train"
validation_data_dir = "/ebs/categories/test"
img_width, img_height = 224, 224
nb_epoch = 30
nb_train_samples = 1430
nb_validation_samples = 500

base_model = VGG19(include_top=False, weights = 'imagenet')
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu', name='fc1')(x)
predictions = Dense(6, activation = 'softmax', name = 'predicts')(x)
model = Model(input= base_model.input, output = predictions)

# Load some top model weights?

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=optimizers.RMSprop(lr = .00001),
            loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = image.ImageDataGenerator(rescale = 1./255,
                                  shear_range =.2,
                                  zoom_range = .2,
                                  horizontal_flip = True)

test_datagen = image.ImageDataGenerator(rescale=1./255)

generator_train = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

generator_test = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(generator_train,
            samples_per_epoch = nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=generator_test,
            nb_val_samples = nb_validation_samples)
