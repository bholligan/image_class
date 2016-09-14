from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np

top_model_weights_path = 'incep_bottleneck_fc_model.h5'

nb_epoch = 50
train_data = np.load(open('/ebs/incep_bottleneck_features_train.npy'))
train_labels = np.array([1] * (5893) + [0] * (14492))

validation_data = np.load(open('/ebs/incep_bottleneck_features_validation.npy'))
validation_labels = np.array([1] * (2523) + [0] * (6210))

inputs = Input(shape=(2048,8,8))
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(inputs)
# add a fully-connected layer
x = Dense(256, activation='relu', name='fc_1')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(input=inputs, output=predictions)

model.compile(optimizer=Adam(lr = .00001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels,
          nb_epoch=nb_epoch, batch_size=32,
          validation_data=(validation_data, validation_labels))

model.save_weights(top_model_weights_path)
