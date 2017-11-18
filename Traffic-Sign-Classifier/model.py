from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda

from time import time
from time import strftime
import numpy as np
import pickle

num_classes = 43
batch_size = 128
epochs = 15
data_augmentation = False

training_file = "traffic-signs-data/train.p"
validation_file= "traffic-signs-data/valid.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']

# Preprocessing

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_valid = to_categorical(y_valid, num_classes)
y_test = to_categorical(y_test, num_classes)

# Data augmentation

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             zca_epsilon=1e-6,
                             rotation_range=10,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0,
                             zoom_range=0.3,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=False,
                             rescale=None,
                             preprocessing_function=None,
                             data_format="channels_last")
# Model
model = Sequential()

# Convolutional
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=x_train[0].shape))
model.add(Conv2D(input_shape=(32, 32, 3),
                 filters=6,
                 kernel_size=(5, 5), strides=(1, 1),
                 padding='valid',
                 dilation_rate=(1, 1),
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Conv2D(filters=16,
                 kernel_size=(5, 5), strides=(1, 1),
                 padding='valid',
                 dilation_rate=(1, 1),
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

# Fully connected
model.add(Flatten())
model.add(Dropout(0.35))
model.add(Dense(units=120))
model.add(Activation('relu'))
model.add(Dense(units=84))
model.add(Activation('relu'))
model.add(Dense(units=num_classes))
model.add(Activation('softmax'))

# Model training
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

# Visualize training
now = strftime("%c")
tensorboard = TensorBoard(log_dir="logs/"+now.format(time()))

if data_augmentation :
    model.fit_generator(datagen.flow(x_train,
                                     y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                        epochs=epochs,
                        workers=4,
                        callbacks=[tensorboard],
                        verbose=0,
                        validation_data=(x_valid, y_valid))
else :
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              callbacks=[tensorboard],
              validation_data=(x_valid, y_valid),
              shuffle=True)

# Model evaluate

loss_and_metrics = model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=0)
print("valid_accuracy" + str(loss_and_metrics))
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("test_accuracy" + str(loss_and_metrics))

model_json = model.to_json()
with open("logs/" + now.format(time()) + "/model.json", "w") as json_file:
    json_file.write(model_json)
