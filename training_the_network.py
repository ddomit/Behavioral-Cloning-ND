import csv
import cv2
import numpy as np
import tensorflow as tf

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = './data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)

print(current_path)
X_train = np.array(images)
y_train = np.array(measurements)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 5, "The number of epochs.")

model = Sequential()
model.add(Lambda(lambda x: x / 255.0, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss ='mse',optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = FLAGS.epochs)

model.save('model.h5')