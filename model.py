import csv
import cv2
import numpy as np
import os
import random

import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda, Cropping2D
from keras.models import load_model
from keras import backend as K

DATA_DIR = './data/'
IMG_DIR = DATA_DIR + '/IMG/'

def load_samples():
  samples = []
  with open(os.path.join(DATA_DIR, './driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      samples.append(line)
  return samples


def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    random.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      angles = []
      for batch_sample in batch_samples:
        name = IMG_DIR+batch_sample[0].split('/')[-1]
        center_image = cv2.imread(name)
        center_angle = float(batch_sample[3])
        images.append(center_image)
        angles.append(center_angle)

      # trim image to only see section with road
      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

#def load_img(src_path):
#  filename = src_path.split('/')[-1]
#  current_path = './data/IMG/' + filename
#  img = cv2.imread(current_path)
#  return img
#
#def load_data():
#  lines = []
#  with open('./data/driving_log.csv') as csvfile:
#    readlines = csv.reader(csvfile)
#    for line in readlines:
#      lines.append(line)
#
#  images = []
#  steering_angles = []
#  for line in lines[1:100]:
#
#    img_center = load_img(line[0])
#    img_left   = load_img(line[1])
#    img_right  = load_img(line[2])
#
#    images.extend([img_center, img_left, img_right])
#
#    # create adjusted steering measurements for the side camera images
#    steering_center = float(line[3])
#    correction = 0.2 # this is a parameter to tune
#    steering_left = steering_center + correction
#    steering_right = steering_center - correction
#    steering_angles.extend([steering_center, steering_left, steering_right])
#
#    # augment data by flipping
#    images.extend([cv2.flip(img_center, 1), cv2.flip(img_left, 1), cv2.flip(img_right, 1)])
#    steering_angles.extend([-steering_center, -steering_left, -steering_right])
#
#  return images, steering_angles

def main():
  model_file = 'model.h5'
  #batch_size = 32
  batch_size = 1
  epochs = 12

  samples = load_samples()
  train_samples, validation_samples = train_test_split(samples, test_size=0.2)

  if K.image_data_format() == 'channels_first':
    print('channels_first')
  else:
    print(K.image_data_format())
  # prepare data
#  _x, _y = load_data()
#  size = len(_x)
#  test_size = int(size*0.1)
#  x_train, x_test = np.array(_x[test_size:]), np.array(_x[:test_size])
#  y_train, y_test = np.array(_y[test_size:]), np.array(_y[:test_size])

# compile and train the model using the generator function
  train_generator = generator(train_samples, batch_size=batch_size)
  validation_generator = generator(validation_samples, batch_size=batch_size)


  # setup model
  input_shape =(160, 320, 3)
  print('Input Shape: {}'.format(input_shape))

  if os.path.exists(model_file):
    print(">> restore model")
    model = load_model(model_file)
  else:
    print(">> model initialize")
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0,0))))
#    model.add(Conv2D(24, kernel_size=(5, 5), sub_sample=(2, 2), activation='relu'))
#    model.add(Conv2D(36, kernel_size=(5, 5), sub_sample=(2, 2), activation='relu'))
#    model.add(Conv2D(48, kernel_size=(5, 5), sub_sample=(2, 2), activation='relu'))
#    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   #model.add(Dropout(0.25))
#   model.add(Dense(128, activation='relu'))
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

#  model.fit(x_train, y_train,
#            batch_size=batch_size,
#            epochs=epochs,
#            verbose=1,
#            shuffle=True,
#            validation_split=0.2)
  model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)//batch_size),
      validation_data=validation_generator, validation_steps=len(validation_samples)//batch_size,
      epochs=epochs)

  score = model.evaluate(x_test, y_test, verbose=1)
  print(model.metrics_names)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  model.save(model_file)

if __name__ == '__main__':
  main()
