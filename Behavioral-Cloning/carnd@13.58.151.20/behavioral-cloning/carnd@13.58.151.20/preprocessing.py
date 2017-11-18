from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, Flatten, Cropping2D, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda

from time import time
from time import strftime
import numpy as np
import pickle

from import_images import load_images_to_pickle

image_limit = 1000

batch_size = 10
input_shape = (160, 320, 3)
pickle_load_dir = "examples/pickle/"

# Import the data

load_images_to_pickle(valid_ratio=0.2, test_ratio=0.2, image_limit=image_limit, pickle_load_dir=pickle_load_dir)

with open(pickle_load_dir + 'train.pickle', 'rb') as f:
    [x_train, y_train] = pickle.load(f)
x_train = np.asarray(x_train)

train_datagen = ImageDataGenerator(featurewise_center=False,
                                   samplewise_center=False,
                                   featurewise_std_normalization=False,
                                   samplewise_std_normalization=False,
                                   zca_whitening=False,
                                   zca_epsilon=1e-6,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0,
                                   zoom_range=0.1,
                                   channel_shift_range=0.,
                                   fill_mode='nearest',
                                   cval=0.,
                                   horizontal_flip=False,
                                   vertical_flip=False,
                                   rescale=None,
                                   preprocessing_function=None,
                                   data_format="channels_last")

# Model
model = Sequential()

# Convolutional
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape))

train_generator = train_datagen.flow(x_train,
                                     y_train,
                                     batch_size=batch_size)



pyplot.imshow(train_generator.next()[0][0].astype(np.uint8))
pyplot.show()