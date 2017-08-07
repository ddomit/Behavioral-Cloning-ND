import csv
import cv2
import numpy as np
import tensorflow as tf

ignoreHeader = False

#Setting Flags so i could change the dataset to train on and the number of epochs, more quickly

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data', 'data', "data file to train on")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")

lines = []
with open('./' + FLAGS.data + '/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
# for line in lines:
# 	source_path = line[0]
# 	filename = source_path.split('/')[-1]
# 	current_path = './data/IMG/' + filename
# 	image = cv2.imread(current_path)
# 	images.append(image)
# 	measurement = float(line[3])
# 	if i == 1:
# 		measurement += 0.2
# 	elif i == 2:
# 		measurement -=  0.2
# 	measurements.append(measurement)

for line in lines:
    if(ignoreHeader):
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = './' + FLAGS.data + '/IMG/' + filename
            image = cv2.imread(current_path)
            images.append(image)
            
            # Add a correction value to the right and left camera images
            measurement = float(line[3])
            if i == 1:
                measurement += 0.2
            elif i == 2:
                measurement -=  0.2
            measurements.append(measurement)
    ignoreHeader = True

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
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


#Model arquitecture


model = Sequential()

# Normalize pixels
model.add(Lambda(lambda x: x / 255.0, input_shape=(160,320,3)))
# Remove extra background noise from the images
model.add(Cropping2D(cropping=((70,25), (0,0))))

#LeNet
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten(input_shape=(160,320,3)))
# model.add(Dense(1))

#Nvidia   -- I tried with various implementations of dropout but at the end, the Nvidia model was the most succesfull
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = "relu"))
# model.add(Dropout(.7))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation = "relu"))
# model.add(Dropout(.7))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = "relu"))
#model.add(Dropout(.5))
model.add(Convolution2D(64,3,3, activation = "relu"))
#model.add(Dropout(.6))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Dropout(.8))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss ='mse',optimizer = 'adam')
print("Fitting the model")

# Start training
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
print("model fitted")
import matplotlib.pyplot as plt



### Print the keys contained in the history object
print(history_object.history.keys())

### Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
model.save('model5.h5')
print("file saved")