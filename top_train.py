import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input

def train_top_model():
    nb_epoch = 50
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([1] * (12901) + [0] * (7622)) #local is 7627

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([1] * (5533) + [0] * (6017)) #local is 6012

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

train_top_model()
