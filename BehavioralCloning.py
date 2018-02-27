### Adapted from Behavioral Cloning: Training the Network
### Assistance from Stack Overflow convert RGB to Gray Scale
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

## Import Steering Angles and Images 
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


def Remove_Excess(y_train,images,max_mag,str_ang_low,str_ang_up): 
    ## Initial histogram and training set characteristics
    unique_values, unique_magnitudes = np.unique(y_train, return_counts = True)
    plt.hist(y_train,bins=np.size(unique_values)) 
    
    ## Remove values that are over represented in the training data
    too_many = np.where(unique_magnitudes > max_mag)
    
    for i in range(0,np.size(too_many)):
        holder = np.where(y_train == unique_values[too_many[0][i]])
        holder_rand = np.random.choice(holder[0],size=(unique_magnitudes[too_many[0][i]] - max_mag))
        holder_sort = np.sort(holder_rand)
        y_train = np.delete(y_train,holder_sort)
        
        for j in range(0,np.size(holder_sort)):
            del images[holder_sort[np.size(holder_sort) - 1 - j]]
    
    ## Remove steering angles below lower bound
    indices_lower = np.where(y_train < str_ang_low)
    y_train = np.delete(y_train,indices_lower)
    
    for i in range(0,np.size(indices_lower)):
        del images[indices_lower[0][np.size(indices_lower) - 1 - i]]
   
    ## Remove steering angles above upper bound
    indices_upper = np.where(y_train > str_ang_up)
    y_train = np.delete(y_train,indices_upper)
    
    for i in range(0,np.size(indices_upper)):
        del images[indices_upper[0][np.size(indices_upper) - 1 - i]]
    
    ## Set X_train to the final image array
    X_train = np.array(images)
    
    ## Final historgram and training set characteristics
    unique_values, unique_magnitudes = np.unique(y_train, return_counts = True)
    plt.hist(y_train,bins=unique_values)
    
    return X_train, y_train

max_mag = 300
str_ang_low = -0.4
str_ang_up  =  0.4
X_train, y_train = Remove_Excess(y_train,images,max_mag,str_ang_low,str_ang_up)

def Convolutional_Network(X_train, y_train):
    ## Convolutional Nerual Network Imports
    from keras.models import Sequential
    from keras.layers import Lambda, Cropping2D
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D

    # Create Convolutional Neural Network
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
    model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=1)
    
    model.save('model.h5')
    return

#Convolutional_Network(X_train,y_train)