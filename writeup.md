# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/self-driving.png "Self Driving in Autonomous Mode"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

As a starting point to make model, I took the NVIDIA model. 
In that model, I added and modified some layers and parameters.
For the first layer, I add Lamda layer to prepare data for training and validating (model.py line 113).
The image_processing() function is for Lamda layer to resize and normalize each image in data.
And the second layer is cropping layer to crop images to speed up training and validating process (model.py line 114).
Then from the third to seventh layer is convolutional layer with various filter and kernel size values (model.py lines 115~119).
These convolution layers use 'ReLU' activation function.
After flatten, there are four fully connected layers.
The summary of my model is as follows.


|Layer (type)            |      Output Shape        |     Param #   |
|:----------------       |:--------------------     | ------------: |
|lambda_1 (Lambda)       |     (None, 80, 160, 3)   |     0         |
|cropping2d_1 (Cropping2D) |   (None, 40, 160, 3)   |     0         |
|conv2d_1 (Conv2D)       |     (None, 18, 78, 24)   |     1824      |
|conv2d_2 (Conv2D)       |     (None, 7, 37, 36)    |     21636     |
|conv2d_3 (Conv2D)       |     (None, 2, 17, 48)    |     43248     |
|conv2d_4 (Conv2D)       |     (None, 1, 16, 64)    |     12352     |
|conv2d_5 (Conv2D)       |     (None, 1, 16, 64)    |     4160      |
|flatten_1 (Flatten)     |     (None, 1024)         |     0         |
|dense_1 (Dense)         |     (None, 100)          |     102500    |
|dense_2 (Dense)         |     (None, 50)           |     5050      |
|dense_3 (Dense)         |     (None, 10)           |     510       |
|dense_4 (Dense)         |     (None, 1)            |     11        |

#### 2. Attempts to reduce overfitting in the model

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

In my model, there is no regularization to reduce overfitting. For iterative training and testing my model, I had applied dropout layer to my model but it did not show the useful effect. Then I searched the way to reduce overfitting and found that more training data is effective. From the simulator, I gathered more data at the second track and I got the expected result.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 127).


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road in both tracks. This training data was resized and normalized at the first step of the model. During training and testing, I used randomly selected images between center, right and left camera. And, for the steering data, I applied a correction value, 0.25. The value was added when the right datum was selected but it was subtracted when the left datum was choosen.
To augment data set, flipped image was applied randomly.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

### Simulation

After training, I run the simulator in autonomous mode with the trained model and data.
Finally, I found the car which drove itself correctly on the road.

![self-driving][image1]
