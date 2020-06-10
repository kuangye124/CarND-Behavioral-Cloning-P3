# Load data using generator
import os
import csv
import cv2
import numpy as np
import sklearn

path = './data/'
samples =[]
with open(path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=128):
    num_samples = len(samples)
    correction = 0.3
    while 1: #loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
            #add center images
                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]
                current_path = path + 'IMG/' + filename
                center_image = cv2.imread(current_path)
                images.append(center_image)
                center_measurement = float(batch_sample[3])
                measurements.append(center_measurement)
            #data augmentation, flip center images horizontally and take 
            #the opposite sign of the steering measurement to help with the left trun bias
                images.append(cv2.flip(center_image,1))
                measurements.append(center_measurement*-1.0)
                
            #add left and right images
                source_path = batch_sample[1]
                filename = source_path.split('/')[-1]
                current_path = path + 'IMG/' + filename
                left_image = cv2.imread(current_path)
                images.append(left_image)
                left_measurement = float(batch_sample[3]) + correction
                measurements.append(left_measurement)
                source_path = batch_sample[2]
                filename = source_path.split('/')[-1]
                current_path = path + 'IMG/' + filename
                right_image = cv2.imread(current_path)
                images.append(right_image)
                right_measurement = float(batch_sample[3]) - correction
                measurements.append(right_measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# PilotNet
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Convolution2D, Flatten, Dense, Dropout

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

model = Sequential()
#set up lambda layer to normalize and mean-center the image, every pixel now ranges from -0.5 to 0.5
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#crop distracting features (skies, trees, etc.)
model.add(Cropping2D(cropping=((70,25),(0,0))))
#implementing NVIDIA Architecture
model.add(Convolution2D(24,(5,5), strides=(2,2), activation="relu"))
model.add(Convolution2D(36,(5,5), strides=(2,2), activation="relu"))
model.add(Convolution2D(48,(5,5), strides=(2,2), activation="relu"))
model.add(Convolution2D(64,(3,3), activation="relu"))
model.add(Convolution2D(64,(3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

# Training and evaluation
from keras.models import Model
import matplotlib.pyplot as plt

model.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
history_object = model.fit_generator(train_generator, steps_per_epoch =
                                     len(train_samples), validation_data = 
                                     validation_generator,
                                     validation_steps = len(validation_samples), 
                                     epochs=10, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')