# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2018_09_24_21_18_09_909.jpg "Center Image"
[image3]: ./examples/left_2018_09_24_21_18_09_909.jpg "Left Image"
[image4]: ./examples/right_2018_09_24_21_18_09_909.jpg "Right Image"
[image5]: ./examples/center_2018_09_24_21_18_42_525.jpg "Original Image"
[image6]: ./examples/flip.jpg "Flip Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* video.mp4 show the test results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5&3x3 filter sizes and depths between 25 and 100 (model.py lines 91-124) 

The model includes ELU layers to introduce nonlinearity (code line 92, etc), and the data is normalized in the model using a Keras lambda layer (code line 89). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 94,98,102,106,110). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 126-129). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 127).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVIDIA network. I thought this model might be appropriate because it was more powerful than LeNet.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I could improve it to avoid overfitting. 

Then I add dropout layers to the network, and train it.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, such the curve after the birdge. To improve the driving behavior in these cases, I add more data about these failed spots .

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 86-124) consisted of a convolution neural network with the following layers and layer sizes ...

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824
_________________________________________________________________
dropout_1 (Dropout)          (None, 31, 158, 24)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 78, 36)        7812
_________________________________________________________________
dropout_2 (Dropout)          (None, 15, 78, 36)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 38, 48)         15600
_________________________________________________________________
dropout_3 (Dropout)          (None, 7, 38, 48)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 36, 64)         27712
_________________________________________________________________
dropout_4 (Dropout)          (None, 5, 36, 64)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 34, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 6528)              0
_________________________________________________________________
dropout_5 (Dropout)          (None, 6528)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               652900
_________________________________________________________________
activation_1 (Activation)    (None, 100)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
activation_2 (Activation)    (None, 50)                0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
activation_3 (Activation)    (None, 10)                0
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================


#### 3. Creation of the Training Set & Training Process

I used the dataset provided by udacity, and add more about some failed spots(especially the curve after the bridge).

An example of data i collected:
![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would add more scenes to enhanced robustness. 
![alt text][image5]
![alt text][image6]


After the collection process, I had 9813 number of data points. I then preprocessed this data by cropped the useless pixels.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3.(I tried epochs 5, but found the mean squared did not reduce after 3.) I used an adam optimizer so that manually training the learning rate wasn't necessary.
