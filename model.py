import os
import csv
import cv2
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []

# Training parameters
START = 0
BATCH_SIZE = 4



with open('../driving_log.csv') as csvfile:
     reader = csv.reader(csvfile)
     for line in reader:
       lines.append(line)


# Splitting the data into validation (20%) and Training data 80%
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(lines, BATCH_SIZE):

  images = []
  measurements = []
  while True: # Looping forever
  
    shuffle(lines)
    sample_lines = lines
    for line in lines:
          # read in images from center, left and right cameras

          source_path = line[0]
          filename = source_path.split('\\')[-1]
          image_center = cv2.imread('../IMG/'+ filename)

          ## images.append(image_center)
          steering_center = float(line[3])

          source_path = line[1]
          filename = source_path.split('\\')[-1]
          filename = filename.strip()
          image_left = cv2.imread('../IMG/'+ filename)

          source_path = line[2]
          filename = source_path.split('\\')[-1]
          filename = filename.strip()         
          image_right = cv2.imread('../IMG/'+ filename)

          # create adjusted steering measurements for the side camera images
          correction = 0.25 # this is a parameter to tune
          
          steering_left = steering_center + correction
          steering_right = steering_center - correction

          # add images and angles to data set
#          images.extend([image_center, image_left, image_right])
#          measurements.extend([steering_center, steering_left, steering_right])
          camera = np.random.choice(['center', 'right', 'left'])
          if camera == 'center':
            images.extend([image_center])
            measurements.extend([steering_center])
          elif camera == 'right':
            images.extend([image_right])
            measurements.extend([steering_right])
          elif camera == 'left':
            images.extend([image_left])
            measurements.extend([steering_left])
          else:
            pass

          ## measurements.append(measurement)
          images, measurements = shuffle(images, measurements)

          ## Converting the images to numpy array
          augmented_images = []
          augmented_measurements = []

          for image, measurement in zip(images, measurements):
            flip_prob = np.random.random()
            if flip_prob > 0.5:
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)
            else:
                augmented_images.append(image)
                augmented_measurements.append(measurement)
            

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)

def image_preproccesing(input):
    from keras.backend import tf as ktf
    output = ktf.image.resize_images(input, (80, 160))
    return (output/255.0-0.5)

## flattened image connecting to simple output node, it will predict

model = Sequential()
model.add(Lambda(image_preproccesing, input_shape=(160, 320, 3), output_shape=(80, 160, 3)))
model.add(Cropping2D(cropping=((30,10),(0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (2,2), activation="relu"))
model.add(Conv2D(64, (1,1), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
 
model.summary()

## Training using the generator function
train_generator      = generator(train_samples, BATCH_SIZE)
validation_generator = generator(validation_samples, BATCH_SIZE)

print('training and validation generation Completed')

sample_epochs = ((len(train_samples) //BATCH_SIZE) * BATCH_SIZE)
print(' length of samples ', sample_epochs)

history_object = model.fit_generator(generator=train_generator, steps_per_epoch=sample_epochs, validation_data=validation_generator, verbose=1, validation_steps=len(validation_samples), epochs=3)

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
print('model.h5 Saved')