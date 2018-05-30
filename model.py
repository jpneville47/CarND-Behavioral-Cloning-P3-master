"""
Adapted from Behavioral Cloning: Training the Network
Assistance from Stack Overflow convert RGB to Gray Scale
Assistance from Stack Overflow randomize two numpy arrays
Plot code used from Udacity Behavioral Cloning: Visualizing Loss
"""
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

"""Import the steering angles and the images for center, left, and right cameras"""
lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
images_processed = []
measurements = []
correction = 0.2

for line in lines:
    """Load Center Image"""
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = filename
    image = plt.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    """Load Left Image"""
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = filename
    image = plt.imread(current_path)
    images.append(image)
    measurement = float(line[3]) + correction
    measurements.append(measurement)
    
    """Load Right Image"""
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = filename
    image = plt.imread(current_path)
    images.append(image)
    measurement = float(line[3]) - correction
    measurements.append(measurement)

y_train = np.array(measurements)

"""Function which removes excess images from the dataset"""
def Remove_Excess(y_train,images,max_mag,str_ang_low,str_ang_up): 
    
    """Randomize the order of the images and steering angles"""
    rng_state = np.random.get_state()
    np.random.shuffle(y_train)
    np.random.set_state(rng_state)
    np.random.shuffle(images)
    
    """Double test images by mirroring - NOT used for final CNN"""
    images_temp = []
    y_temp = []
    for i in range(0,np.size(y_train)):
        images_temp.append(cv2.flip(images[i],1))
        y_temp.append(-(y_train[i]))
    
    #y_train = np.concatenate((y_train,y_temp),axis=0)
    #images = images + images_temp
    
    """Initial histogram and training set characteristics"""
    unique_values, unique_magnitudes = np.unique(y_train, return_counts = True)
    
    """Remove values that are over represented in the training data"""
    too_many = np.where(unique_magnitudes > max_mag)
    
    for i in range(0,np.size(too_many)):
        holder = np.where(y_train == unique_values[too_many[0][i]])
        y_train = np.delete(y_train,holder)
        
        for j in range(0,np.size(holder)):
            del images[holder[0][np.size(holder) - 1 - j]]
    
    """Remove steering angles below lower bound"""
    indices_lower = np.where(y_train < str_ang_low)
    y_train = np.delete(y_train,indices_lower)
    
    for i in range(0,np.size(indices_lower)):
        del images[indices_lower[0][np.size(indices_lower) - 1 - i]]
   
    """Remove steering angles above upper bound"""
    indices_upper = np.where(y_train > str_ang_up)
    y_train = np.delete(y_train,indices_upper)
    
    for i in range(0,np.size(indices_upper)):
        del images[indices_upper[0][np.size(indices_upper) - 1 - i]]
    
    """Set X_train to the final image array"""
    X_train = np.array(images)
    
    """Final historgram and training set characteristics"""
    unique_values, unique_magnitudes = np.unique(y_train, return_counts = True)
    plt.figure(1)
    plt.hist(y_train,bins=unique_values)
    
    return X_train, y_train

"""Set the maximum number of images allowed for a given steering angle"""
max_mag = 200

"""Set the limit for negative steering angles"""
str_ang_low = -1

"""Set the limit for positive steering angles"""
str_ang_up  =  1

"""
Remove the excess images from the dataset using max_mag, str_ang_low, and
str_ang_up as the limits. Return the finalized dataset to be trained.
"""
X_train, y_train = Remove_Excess(y_train,images,max_mag,str_ang_low,str_ang_up)

"""Convolutional neural network for generating a steering angle based on an image"""
def Convolutional_Network(X_train, y_train):
    ## Convolutional Nerual Network Imports
    from keras.models import Sequential
    from keras.layers import Lambda, Cropping2D
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D

    """Create Convolutional Neural Network"""
    model = Sequential()
    """Normalize the matrix values"""
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
    """
    Crop the image to ignore pieces of the environment that do not offer
    information gain.
    """
    model.add(Cropping2D(cropping=((70,24),(60,60))))
    model.add(Convolution2D(24,5,strides=(2,2)))
    model.add(Dropout(0.3))
    model.add(Convolution2D(36,5,strides=(2,2)))
    model.add(Dropout(0.3))
    model.add(Convolution2D(48,5,strides=(2,2)))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64,3,strides=(1,1)))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64,3,strides=(1,1)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100,  activation='relu'))
    model.add(Dense(50,   activation='relu'))
    model.add(Dense(10,   activation='relu'))
    model.add(Dense(1))
    
    """
    Compile and Fit the network and return the history for the validatio and 
    training loss functions.
    """
    model.compile(loss='mse',optimizer='adam')
    history = model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=2)
    
    model.save('model.h5')
    return history

"""Create Netural Network and return history of loss function per epoch"""
history = Convolutional_Network(X_train,y_train)  

"""plot the training and validation loss for each epoch"""
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()