import os
import csv
from keras.models import Sequential, Model
from keras.layers import Cropping2D,Flatten,Lambda,Dense,Activation,Dropout,MaxPooling2D
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from scipy import ndimage


samples = []
file_dir = "/opt/carnd_p3/data"
#file_dir = "/home/workspace/data"
with open(file_dir+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1::]


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            correction = 0.1 # this is a parameter to tune
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    #name = './IMG/'+batch_sample[i].split('/')[-1]
                    #name = file_dir+'/IMG/'+batch_sample[i].split('/')[-1]
                    if batch_sample[i].find('\\') != -1 :
                        name = file_dir+'/IMG/'+batch_sample[i].split('\\')[-1]
                    else:
                        name = file_dir+'/IMG/'+batch_sample[i].split('/')[-1]                    
                    #center_image = cv2.imread(name)
                    center_image = ndimage.imread(name)
                    
                    center_angle = float(batch_sample[3])
                    
                    if i == 0:
                        images.append(center_image)
                        angles.append(center_angle)
                        images.append(cv2.flip(center_image,1))
                        angles.append(center_angle*-1.0)
                    elif i ==1:
                        center_angle = center_angle+correction
                        images.append(center_image)
                        angles.append(center_angle)
                        images.append(cv2.flip(center_image,1))
                        angles.append(center_angle*-1.0)
                    else:
                        center_angle = center_angle-correction
                        images.append(center_image)
                        angles.append(center_angle)
                        images.append(cv2.flip(center_image,1))
                        angles.append(center_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format
activate_func = 'elu'

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: ((x / 255.0) - 0.5), input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24,(5,5),strides=(2,2),activation=activate_func)) 
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(36,(3,3),strides=(2,2),activation=activate_func))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(48,(3,3),strides=(2,2),activation=activate_func))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3),activation=activate_func))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3),activation=activate_func))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation(activate_func))
#model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation(activate_func))
#model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation(activate_func))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.summary()

model.compile(optimizer=Adam(0.0001), loss="mse")
#model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,  nb_val_samples=len(validation_samples), nb_epoch=3)

"""
If the above code throw exceptions, try 
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
"""
model.save('model.h5')  # creates a HDF5 file 'model.h5'



