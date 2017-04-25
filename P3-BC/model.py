# Udacity Self-Driving Car Nanodegree
#
# Project 3 - Behavioral Cloning
# Matt Blomquist
#
#
# Additional Information: See README.md @: https://github.com/mblomquist/Udacity-SDCND-Behavioral-Cloning-P3

# Import Libraries and Modules
import keras
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from sklearn.utils import shuffle
import cv2
from sklearn.model_selection import train_test_split

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

# ------------------------------------------
# ------------ Import and Process Data -----
# ------------------------------------------

# Import Training Data from CSV File
import csv

# Main Driving Log
data_c = csv.reader(open('data_log/center_driving_log.csv'), delimiter=",",quotechar='|')
data_l = csv.reader(open('data_log/left_driving_log.csv'), delimiter=",",quotechar='|')
data_r = csv.reader(open('data_log/right_driving_log.csv'), delimiter=",",quotechar='|')


# Create Empty Arrays for Center Driving Log Data
img_center = []
img_left = []
img_right = []
steering = []

# Create Empty Arrays for Left Driving Log Data
img_center_lc = []
img_left_ll = []
img_right_lr = []
steering_l = []

# Create Empty Arrays for Right Driving Log Data
img_center_rc = []
img_left_rl = []
img_right_rr = []
steering_r = []

# Populate Center Driving Arrays
for row in data_c:
      img_center.append(row[0])
      img_left.append(row[1])
      img_right.append(row[2])
      steering.append(row[3])

      # Populate Left Driving Arrays
for row in data_l:
      img_center_lc.append(row[0])
      img_left_ll.append(row[1])
      img_right_lr.append(row[2])
      steering_l.append(row[3])

# Populate Right Driving Arrays
for row in data_r:
      img_center_rc.append(row[0])
      img_left_rl.append(row[1])
      img_right_rr.append(row[2])
      steering_r.append(row[3])

# Center Array
img_center = np.asarray(img_center)
img_left = np.asarray(img_left)
img_right = np.asarray(img_right)

steering = np.asarray(steering, dtype=np.float32)

# Left Array (Add Bias)
bias_left = .2
img_center_lc = np.asarray(img_center_lc)
img_left_ll = np.asarray(img_left_ll)
img_right_lr = np.asarray(img_right_lr)

steering_l = np.asarray(steering_l, dtype=np.float32)+bias_left

# Right Array (Add Bias)
bias_right = -.2
img_center_rc = np.asarray(img_center_rc)
img_left_rl = np.asarray(img_left_rl)
img_right_rr = np.asarray(img_right_rr)

steering_r = np.asarray(steering_r, dtype=np.float32)+bias_right

# Concatenate img and label lists
img_list = np.concatenate((img_center, img_left, img_right, img_center_lc, img_left_ll, img_right_lr, img_center_rc, img_left_rl, img_right_rr), axis=0)
label_list = np.concatenate((steering, steering, steering, steering_l, steering_l, steering_l, steering_r, steering_r, steering_r), axis=0)

# Shuffle X_train and y_train Data
img_list, label_list = shuffle(img_list, label_list)

# Split Validation Set
img_list, img_val, label_list, label_val = train_test_split(img_list, label_list, test_size=.1)

# ------------------------------------------
# ------------ Preprocess Data  ------------
# ------------------------------------------


# Adjust the Brightness of the Image
def adjust_brightness(image):

      image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
      image[:,:,2] = image[:,:,2]*(.25+np.random.uniform())
      image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)

      return image
                      
# Create function for X_train data normalization
def normalize_grayscale(image_data):
    a = -1
    b = 1
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

# Rescale Image
def rescale_image(image):

    image = image[32:135, 0:320]
    image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)    
    return image

# Translate Image
def transImage(image,label):

      x = np.random.randint(-20,20)
      y = np.random.randint(-20,20)

      M = np.float32([[1,0,x],[0,1,y]])

      image = cv2.warpAffine(image,M,(320,160))
      label = label+(-1*x*.002)

      return image, label

# Flip Image
def flip_image(image,label):

      image = cv2.flip(image,1)
      label = -1*label

      return image, label

# Create Generator for Keras
def data_gen(batch_size, img_list, label_list):

      # Set up Empty Batches
      batch_image = np.zeros((batch_size, 64, 64, 3))
      batch_label = np.zeros(batch_size)

      j = 0

      # Shuffle Data for Diversity
      img_list, label_list = shuffle(img_list, label_list)

      # Run batch yield loop
      while 1:

            i = 0

            # Define Batch Start and End for Data
            batch_start = batch_size*j
            batch_stop = batch_size*(j+1)-1

            # Check End Batch Stop Size
            if batch_stop >= len(img_list)-1:
                  batch_stop = len(img_list)-1

                  j = 0

            # Generate Data
            for batch in range(batch_start,batch_stop,1):

                  img_new = cv2.imread(img_list[batch])
                  label_new = np.float32(label_list[batch])

                  # Run preprocess pipeline
                  img_new, label_new = transImage(img_new,label_new)

                  img_new = adjust_brightness(img_new)

                  img_new = rescale_image(img_new)

                  if batch % 3 == 0:

                        img_new, label_new = flip_image(img_new,label_new)

                  batch_image[i] = img_new
                  batch_label[i] = label_new

                  i = i+1

            j = j+1

            # Output Batch
            yield (batch_image, batch_label)


# ------------------------------------------
# ------------ Build and Train Model -------
# ------------------------------------------

# Import Keras Libraries to be used.
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

# Define Keras model architecture with Sequential
model = Sequential()

# Input Convolutional Layer - I/O Match to Use Best Color Space 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(64,64,3)))

# Add Non-Linearity
model.add(Activation('elu'))

# Add Convolutional Layer Set - 5x5 filter kernal with 2x2 strides (nVidia Architecture)
model.add(Convolution2D(3, 5, 5, subsample=(2, 2), border_mode="valid", name='layer1_conv2d'))

# Add Non-Linearity
model.add(Activation('elu'))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", name='layer2_conv2d'))

# Add Non-Linearity
model.add(Activation('elu'))

model.add(Convolution2D(36, 5, 5, subsample=(1, 1), border_mode="valid", name='layer3_conv2d'))

# Add Dropout to Reduce Over-fitting
model.add(Dropout(.5))

# Add Non-Linearity
model.add(Activation('elu'))

# Add Convolutional Layer Set - 3x3 filter kernal with 1x1 strides (nVidia Architecture)
model.add(Convolution2D(48, 3, 3, subsample=(1, 1), border_mode="valid", name='layer4_conv2d'))

# Add Non-Linearity
model.add(Activation('elu'))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", name='layer5_conv2d'))

# Add Dropout to Reduce Over-fitting
model.add(Dropout(.5))

# Add Non-Linearity
model.add(Activation('elu'))

# Convert to Fully Connected Layer
model.add(Flatten())

# Add Dropout to Reduce Over-fitting
model.add(Activation('elu'))

# Add Dense Layer Set - 100 > 50 > 10 (nVidia Architecture)
model.add(Dense(100))
model.add(Dropout(.5))
model.add(Activation('elu'))

model.add(Dense(50))
model.add(Activation('elu'))

model.add(Dense(10))

# Reduce to Single Output / Prediction Value - Use tanh Activation to bound results between -1 and 1.
model.add(Dense(1))

# Print Model Summary
print(model.summary())

# Set Compiler and Loss function for Keras Model
# Note: See README for additional information.
model.compile(optimizer='adam', loss='mse')

# Train and Validate Keras Model with X_train, y_train
# Note: See README for additional information and justification
model.fit_generator(data_gen(512, img_list, label_list), samples_per_epoch=len(img_list), nb_epoch=3, validation_data=data_gen(512, img_val, label_val), nb_val_samples=len(img_val))

# Test on 10 New Images / Output Predictions
img_test = np.zeros((10,64,64,3))

for i in range(0,10,1):

      file_name = 'data_log/test/test_0'+str(i)+'.jpg'

      img_new = cv2.imread(file_name)
      img_new = adjust_brightness(img_new)
      img_new = rescale_image(img_new)

      img_test[i] = img_new

preds = model.predict(img_test, batch_size=10, verbose=0)
print(preds)

# ------------------------------------------
# ------- Output Keras Model to File -------
# ------------------------------------------

# Output Keras Model in .json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Output Keras Model Weights
model.save_weights("model.h5")
print("Saved model to disk")
