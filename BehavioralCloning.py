### Adapted from Behavioral Cloning: Training the Network
### Assistance from Stack Overflow convert RGB to Gray Scale
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
images_processed = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = filename
    image = plt.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

y_train = np.array(measurements)
    
unique_values, unique_magnitudes = np.unique(y_train, return_counts = True)
plt.hist(y_train,bins=np.size(unique_values)) 

# Remove excess data
too_many = np.where(unique_magnitudes > 300)
indices = []

#for i in range(0,np.size(too_many)):
#    holder = np.where(y_train == too_many[0][i])
#    indices = np.concatenate((indices,holder), axis=0)

indices = np.where(y_train < -0.4)
y_train = np.delete(y_train,indices)

for i in range(0,np.size(indices)):
    del images[indices[0][np.size(indices) - 1 - i]]
    
X_train = np.array(images)

unique_values, unique_magnitudes = np.unique(y_train, return_counts = True)
plt.hist(y_train,bins=unique_values)

# Create Convolutional Neural Network
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,24),(60,60))))
model.add(Convolution2D(24,5,strides=(2,2)))
model.add(Convolution2D(36,5,strides=(2,2)))
model.add(Convolution2D(48,5,strides=(2,2)))
model.add(Convolution2D(64,3,strides=(1,1)))
model.add(Convolution2D(64,3,strides=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=3)

model.save('model.h5')

