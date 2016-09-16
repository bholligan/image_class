from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD, RMSprop

train_data_dir = "/ebs/categories/train"
validation_data_dir = "/ebs/categories/test"
img_width, img_height = 299, 299
nb_epoch = 25
nb_train_samples = 9951
nb_validation_samples = 4777

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu', name='fc_1')(x)
predictions = Dense(13, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=SGD(lr = .00001), loss = 'categorical_crossentropy', metrics=['accuracy'])

train_datagen = image.ImageDataGenerator(rescale = 1./255,
                                  shear_range =.2,
                                  zoom_range = .2,
                                  horizontal_flip = True)

# Inception has a custom image preprocess function
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
            nb_epoch = nb_epoch,
            validation_data = generator_test,
            nb_val_samples = nb_validation_samples)

#start fine-tuning
# train the top 2 inception blocks
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
            loss='categorical_crossentropy', metrics=['accuracy'])

# # we train our model again (this time fine-tuning the top 2 inception blocks
# # alongside the top Dense layers
model.fit_generator(generator_train,
            samples_per_epoch = nb_train_samples,
            nb_epoch = nb_epoch,
            validation_data = generator_test,
            nb_val_samples = nb_validation_samples)

model_json = model.to_json()
with open("incep_multi.json", 'w') as json_file:
    json_file.write(model_json)
model.save_weights("incep_multi.h5")
