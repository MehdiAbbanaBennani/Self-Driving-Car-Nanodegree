from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, Flatten, Cropping2D, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda
from keras.backend import tf as ktf

from time import time
from time import strftime
import numpy as np
import pickle

from import_images import load_images_to_pickle

image_limit = 2000
load_images = True
center_only = True
steering_correction = 0.2 # To tune

batch_size = 128
epochs = 15
input_shape = (160, 320, 3)
pickle_load_dir = "examples/pickle/"
now = strftime("%c")
log_dir = "logs/" + now.format(time())

# Import the data
if load_images :
    load_images_to_pickle(valid_ratio=0.1, test_ratio=0.1,
                      image_limit=image_limit, steering_correction=steering_correction,
                      pickle_load_dir=pickle_load_dir)

if center_only:
    pickle_load_dir = "examples/pickle/center_angle/"
else:
    pickle_load_dir = "examples/pickle/all_angles/"

with open(pickle_load_dir + 'train.pickle', 'rb') as f:
    [x_train, y_train] = pickle.load(f)
with open(pickle_load_dir + 'valid.pickle', 'rb') as f:
    [x_valid, y_valid] = pickle.load(f)
with open(pickle_load_dir + 'test.pickle', 'rb') as f:
    [x_test, y_test] = pickle.load(f)

train_size = x_train.shape[0]
valid_size = x_valid.shape[0]
test_size = x_test.shape[0]

# Data augmentation

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
                                   zoom_range=0,
                                   channel_shift_range=0.,
                                   fill_mode='nearest',
                                   cval=0.,
                                   horizontal_flip=False,
                                   vertical_flip=False,
                                   rescale=None,
                                   preprocessing_function=None,
                                   data_format="channels_last")

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
del x_train
valid_generator = test_datagen.flow(x_valid, y_valid, batch_size=batch_size)
del x_valid
test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)
del x_test

# Model
model = Sequential()

# Convolutional
model.add(Cropping2D(cropping=((45, 5), (0, 0)), input_shape=input_shape))
Lambda(lambda image: ktf.image.resize_images(image, (80, 200)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Conv2D(filters=6,
                 kernel_size=(5, 5), strides=(2, 2),
                 padding='valid',
                 dilation_rate=(1, 1),
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'))
model.add(Activation('relu'))

model.add(Conv2D(filters=16,
                 kernel_size=(5, 5), strides=(2, 2),
                 padding='valid',
                 dilation_rate=(1, 1),
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'))
model.add(Activation('relu'))

model.add(Conv2D(filters=32,
                 kernel_size=(5, 5), strides=(2, 2),
                 padding='valid',
                 dilation_rate=(1, 1),
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'))
model.add(Activation('relu'))

model.add(Conv2D(filters=32,
                 kernel_size=(3, 3), strides=(1, 1),
                 padding='valid',
                 dilation_rate=(1, 1),
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'))
model.add(Activation('relu'))

model.add(Conv2D(filters=32,
                 kernel_size=(3, 3), strides=(1, 1),
                 padding='valid',
                 dilation_rate=(1, 1),
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'))
model.add(Activation('relu'))

# Fully connected
model.add(Flatten())
# model.add(Dropout(0.35))
model.add(Dense(units=1164))
model.add(Activation('relu'))
model.add(Dense(units=100))
model.add(Activation('relu'))
model.add(Dense(units=50))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('relu'))
model.add(Dense(units=1))
# model.add(Lambda(lambda x: np.sign(x) * np.max(abs(x), 90)))

# Model training

model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['mae'])

# Visualize training
tensorboard = TensorBoard(log_dir=log_dir)

# Train the model
model.fit_generator(train_generator,
                    steps_per_epoch=int(np.ceil(train_size / float(batch_size))),
                    epochs=epochs,
                    workers=4,
                    callbacks=[tensorboard],
                    verbose=1,
                    validation_data=valid_generator,
                    validation_steps=int(np.ceil(valid_size / float(batch_size))))

# Evaluate the model
loss_and_metrics = model.evaluate_generator(valid_generator, steps=int(np.ceil(valid_size / float(batch_size))))
print("valid_mae" + str(loss_and_metrics))
loss_and_metrics = model.evaluate_generator(test_generator, steps=int(np.ceil(valid_size / float(batch_size))))
print("test_mae" + str(loss_and_metrics))

# Saving the model
model_json = model.to_json()
with open(log_dir + "/model.json", "w") as json_file:
    json_file.write(model_json)

model.save(log_dir + '/my_model.h5')
del model