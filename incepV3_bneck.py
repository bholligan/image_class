from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adam

train_data_dir = "/ebs/user05/data/train"
validation_data_dir = "/ebs/user05/data/test"
img_width, img_height = 299, 299
nb_train_samples = 20385
nb_validation_samples = 8733

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

datagen = image.ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)
bottleneck_features_train = base_model.predict_generator(generator, nb_train_samples)
np.save(open('incep_bottleneck_features_train.npy', 'w'), bottleneck_features_train)

generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)
bottleneck_features_validation = base_model.predict_generator(generator, nb_validation_samples)
np.save(open('incep_bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
