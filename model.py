import csv
import cv2
import numpy as np

import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda, Cropping2D
from keras.models import load_model
from keras import backend as K

def load_img(src_path):
  filename = src_path.split('/')[-1]
  current_path = './data/IMG/' + filename
  img = cv2.imread(current_path)
  return img

def load_data():
  lines = []
  with open('./data/driving_log.csv') as csvfile:
    readlines = csv.reader(csvfile)
    for line in readlines:
      lines.append(line)

  images = []
  steering_angles = []
  for line in lines[1:100]:

    img_center = load_img(line[0])
    img_left   = load_img(line[1])
    img_right  = load_img(line[2])

    images.extend([img_center, img_left, img_right])

    # create adjusted steering measurements for the side camera images
    steering_center = float(line[3])
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    steering_angles.extend([steering_center, steering_left, steering_right])

    # augment data by flipping
    images.extend([cv2.flip(img_center, 1), cv2.flip(img_left, 1), cv2.flip(img_right, 1)])
    steering_angles.extend([-steering_center, -steering_left, -steering_right])

  return images, steering_angles

def main():
  model_file = 'model.h5'
  batch_size = 1
  epochs = 12

  # prepare data
  _x, _y = load_data()
  size = len(_x)
  test_size = int(size*0.1)
  x_train, x_test = np.array(_x[test_size:]), np.array(_x[:test_size])
  y_train, y_test = np.array(_y[test_size:]), np.array(_y[:test_size])


  # setup model
  input_shape = _x[0].shape
  print('Input Shape: {}'.format(input_shape))

  if os.path.exists(model_file):
    print(">> restore model")
    model = load_model(model_file)
  else:
    print(">> model initialize")
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0,0))))
    model.add(Conv2D(24, kernel_size=(5, 5), sub_sample=(2, 2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), sub_sample=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), sub_sample=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   #model.add(Dropout(0.25))
#   model.add(Dense(128, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

  model.compile(loss="mean_squared_error",
                optimizer='adam',
                metrics=['accuracy'])

  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_split=0.2)

  score = model.evaluate(x_test, y_test, verbose=1)
  print(model.metrics_names)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  model.save(model_file)

if __name__ == '__main__':
  main()
