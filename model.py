import csv
import cv2
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Reshape, Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda, Cropping2D
from keras.models import load_model
from keras import backend as K
from keras.layers.normalization import BatchNormalization
DATA_DIR = './DATA/'
IMG_SUBDIR = '/IMG/'

def load_samples():
  samples = []
  for dirpath in os.listdir(DATA_DIR):
    print(dirpath)
    try:
      with open(os.path.join(DATA_DIR + dirpath, './driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        # skip fitst line
        for i, line in enumerate(reader):
          if i > 0:
            samples.append((DATA_DIR + dirpath, line))
    except Exception as e:
      print(e)
  return samples

def load_img(dirpath, filepath):
  filename = filepath.split('/')[-1]
  path = os.path.join(dirpath + IMG_SUBDIR, filename)
  img = cv2.imread(path)
  if img == None:
    print(path)
    sys.exit()
  return img

def generator(samples, batch_size=32, augment_factor=6):
  correction = 0.15 # this is a parameter to tune
  num_samples = len(samples)

  while 1: # Loop forever so the generator never terminates
    random.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      angles = []
      for batch_sample in batch_samples:
        dirpath, sample = batch_sample

        center_image = load_img(dirpath, sample[0])
        left_image   = load_img(dirpath, sample[1])
        right_image  = load_img(dirpath, sample[2])
        center_angle = float(sample[3])
        left_angle  = center_angle + correction
        right_angle = center_angle - correction

        images.extend([center_image, left_image, right_image])
        angles.extend([center_angle, left_angle, right_angle])

        # flip
        images.extend([cv2.flip(center_image, 1), cv2.flip(left_image, 1), cv2.flip(right_image, 1)])
        angles.extend([-center_angle, -left_angle, -right_angle])

      # trim image to only see section with road
      X_train = np.array(images)
      y_train = np.array(angles)
      ret = sklearn.utils.shuffle(X_train, y_train)
      for i in range(augment_factor):
        offset = batch_size*i
        yield (ret[0][offset:offset + batch_size], ret[1][offset:offset + batch_size])

def main():
  model_file = 'model.h5'
  batch_size = 192
  augment_factor=6
  epochs = 5

  # prepare data
  samples = load_samples()
  random.seed(0)
  random.shuffle(samples)
  train_samples, validation_samples = train_test_split(samples, test_size=0.2)

  if K.image_data_format() == 'channels_first':
    print('channels_first')
  else:
    print(K.image_data_format())

  # set generator function
  train_generator = generator(train_samples, batch_size=batch_size)
  validation_generator = generator(validation_samples, batch_size=batch_size)

  print("num_samples:{}".format(len(samples)))
  print("num_batches:{}".format(len(samples)//batch_size))

  # setup model
  input_shape =(160, 320, 3)
  reshape_shape =(160, 320, 3)
  print('Input Shape: {}'.format(input_shape))

  if os.path.exists(model_file):
    print(">> restore model")
    model = load_model(model_file)
  else:
    print(">> model initialize")
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
#    model.add(BatchNormalization())
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))

  model.compile(loss="mean_squared_error",
                optimizer='adam',
                metrics=['accuracy'])

  history_object = model.fit_generator(train_generator, steps_per_epoch=(augment_factor*len(train_samples)//batch_size),
      validation_data=validation_generator, validation_steps=len(validation_samples)//batch_size,
      epochs=epochs)

  # save model
  model.save(model_file)
  print('saving model done.')

  ### print the keys contained in the history object
  print(history_object.history.keys())
  ### plot the training and validation loss for each epoch
  plt.plot(history_object.history['loss'])
  plt.plot(history_object.history['val_loss'])
  plt.title('model mean squared error loss')
  plt.ylabel('mean squared error loss')
  plt.xlabel('epoch')
  plt.legend(['training set', 'validation set'], loc='upper right')
  plt.savefig('visualization.png')

  print('saving loss visualization done.')

if __name__ == '__main__':
  main()
