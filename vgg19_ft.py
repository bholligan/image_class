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
from keras.layers import GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.models import Model

top_model_weights_path = 'vgg19_fc_model_ft.h5'
train_data_dir = "/ebs/user05/data/train"
validation_data_dir = "/ebs/user05/data/test"
img_width, img_height = 224, 224
nb_epoch = 20
nb_train_samples = 20523
nb_validation_samples = 11550

base_model = VGG19(include_top=False, weights = 'imagenet')
x = base_model.output
x = Flatten(input_shape = (512,7,7), name='flatten')(x)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation = 'sigmoid', name = 'predicts')(x)
model = Model(input= base_model.input, output = predictions)

# Load some top model weights?

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=optimizers.RMSprop(lr = .000001),
            loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = image.ImageDataGenerator(rescale = 1./255,
                                  shear_range =.2,
                                  zoom_range = .2,
                                  horizontal_flip = True)

test_datagen = image.ImageDataGenerator(rescale=1./255)

generator_train = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

generator_test = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

model.fit_generator(generator_train,
            samples_per_epoch = nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=generator_test,
            nb_val_samples = nb_validation_samples)
