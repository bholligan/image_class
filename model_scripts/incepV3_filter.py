from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adam

train_data_dir = "/ebs/user05/data/train"
validation_data_dir = "/ebs/user05/data/test"
img_width, img_height = 299, 299
nb_epoch = 20
nb_train_samples = 20385
nb_validation_samples = 8733

# Create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer
x = Dense(256, activation='relu', name='fc_1')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(input=base_model.input, output=predictions)

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=RMSprop(lr = .00001), loss = 'binary_crossentropy', metrics=['accuracy'])

train_datagen = image.ImageDataGenerator(rescale = 1./255,
                                  shear_range =.2,
                                  zoom_range = .2,
                                  horizontal_flip = True)

# Rescale image color scale
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
            nb_epoch = nb_epoch,
            validation_data = generator_test,
            nb_val_samples = nb_validation_samples)

model_json = model.to_json()
with open("incep_filter.json", 'w') as json_file:
    json_file.write(model_json)
model.save_weights("incep_filter.h5")
