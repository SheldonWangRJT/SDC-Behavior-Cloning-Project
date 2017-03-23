import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader (csvfile)
    for line in reader:
        lines.append(line)

print("===========Printing a line of the csv data: ===========")
print(lines[0])

images = []
measurements = []

for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = './data/IMG/' + filename
 
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        
        #added multi-camera correction factor
        correction_factor = 0.2
        
        if i==0:
            measurement = measurement
        elif i==1:
            measurement += correction_factor
        else:
            measurement -= correction_factor
        
        measurements.append(measurement)
    
        images.append(cv2.flip(image,1))
        measurements.append(-1*measurement)
    

print("===========Printing the first steering wheel measurement data: ===========")    
print(measurements[0])
print("===========Printing the first center image path string:===========")
print(lines[0][0])
print("===========Printing the first center image dimension/shape:===========")
img = cv2.imread('.\data\IMG\center_2017_03_16_22_45_30_515.jpg')
print(img.shape)

###Until now we are appending all the elements from the csv table into our python arrays

print(len(images),"*3 images have been saved")

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#Create the training data set
X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
#Preprocess with normalization
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20), (0,0))))

#Add a switch for different solutions option = 1 is using LeNet,
#------------------------------------ option = 2 is NVidia Model

model_option = 2

if model_option == 1:
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    
elif model_option == 2:
    model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu"))
    model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu"))
    model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu"))
    model.add(Convolution2D(64,(1,1),strides=(2,2),activation="relu"))
    #model.add(Convolution2D(64,3,3,subsample=(2,2),activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(0.1))
    model.add(Dense(10))
    model.add(Dense(1))

#Use mean square error to optimize the network with adam optimizer
model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)
model.save('model.h5')
print("data saved")
