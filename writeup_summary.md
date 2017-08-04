#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model-visualization.png "Model Visualization"
[image2]: ./examples/center_lane.png "Driving on the center of the lane"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* training_the_network_3images.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model2.h5 containing a trained convolution neural network 
* writeup_summary.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track for more than 3 laps by executing 
```sh
python drive.py model.h5
```

or 

```sh
python ./drive.py model2.h5 Video > /dev/null
```
For a non verbose, more clean implementation

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 324 and 64 (model.py lines 94-98) 

The model includes RELU layers to introduce nonlinearity (code line 102 to 104), and the data is normalized in the model using a Keras lambda layer (code line 81). 

The model also includes a Cropping of the images to help with classification and removing noise (code line 83)

####2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 103). I actually tried using more dropout layers but more didnt seem to be better. 

The model was trained and validated on different data sets with model.fit to ensure that the model was not overfitting (code line 114). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 114).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and I drove around the track backwards to ensure that it wasnt just learning the track

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to go back to the lessons where different methods had been described.

My first step was to use a convolution neural network model similar to the LeNet arquitechture. I thought this model might be appropriate because it runs convolutional nerual networks which are very good at identifying images, but then I realized that the Nvidia arquitechture was better for the task at hand.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that a dropout layer would prevent this, I actually ran various dropout layers to see how this impacted the model but I realized at some point it starts to be worse.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track so to improve the driving behavior in these cases, I went back to the simulator to drive over this places again and again untill the model had enought data to steer in the right direction.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road for more than 3 laps.

####2. Final Model Architecture

The final model architecture (model.py lines 78-111) consisted of a convolution neural network with the following layers and layer sizes 3 convolutional layers with 2by2 filters 2 convolutional layers with relu activation, then a Dropout layer Followed by a Flatten layer and to finish with 4 dense layers going from 100 to 1 output.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]



I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
