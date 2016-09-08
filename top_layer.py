from keras.preprocessing import image
import numpy as np
from keras.applications import imagenet_utils
import json
from keras.applications.imagenet_utils import *
import pandas as pd
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.utils.data_utils import get_file
import os
import h5py

# img dimensions
img_width, img_height = 224, 224

# path to the model weights file.
weights_file = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
weights_path = get_file('vgg19_weights_th_dim_ordering_th_kernels_notop.h5',
                                        weights_file,
                                        cache_subdir='models')

train_data_dir = "/ebs/user05/data/train"
test_data_dir = "/ebs/user05/data/test"
nb_train_samples = 20528
nb_validation_samples = 11545
nb_epoch = 50

def save_bottleneck_features():
    # Look into subtracting out the mean pixel value instead of rescaling
    datagen = image.ImageDataGenerator(rescale=1./255)

    # build the VGG19 network
    model = Sequential()
    # Block 1
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',
                            input_shape=(3,224,224), name='block1_conv1'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.load_weights(weights_path)

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)

save_bottleneck_features()
