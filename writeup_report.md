# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./visualization.png "Visualization"
[image1]: ./model.png "Model Visualization"
[image2]: ./image2.jpg "Grayscaling"
[image3]: ./image3.jpg "Recovery Image"
[image4]: ./image4.jpg "Recovery Image"
[image5]: ./image5.jpg "Recovery Image"
[image6]: ./normal.jpg "Normal Image"
[image7]: ./flipped.jpg "Flipped Image"
[image8]: ./center.jpg "Right Image"
[image9]: ./left.jpg "Left Image"
[image10]: ./right.jpg "Right Image"
[image11]: ./track2_0.jpg "Track2 Image0"
[image12]: ./track2_1.jpg "Track2 Image1"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 A video recording of your vehicle driving autonomously at least one lap around the track

**Final Result:** [video.mp4](./video.mp4), I recorded two laps 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 3 ConvNet with 5x5 filter sizes and depths between 24, 36, 48 and 2 additional ConvNet with 3x3 filters sizes and depths between 64 and 64 (model.py lines 118-133) 

The model includes RELU layers to introduce nonlinearity after every ConvNet, and the data is normalized in the model using a Keras lambda layer (code line 120). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting after the first 2 fully-connected layers, not before the output layer (model.py lines 129, 131). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 139-141). I used fit_generator function for large size input images. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 135).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also get a few images from the second track. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I decided to follow NVIDIA configurations, because the model is complicated enough to capture geometrical features from the collected images for the project. And I focus on Data collection and Data Augmentations.

I increased data after every experiments. Whenever I ran trainig mode I collected more than 2000 images. And I focued on collecting images on the curves of the road. Because in the first few experiments my model failed to go properly on the curves ,and go away out of lanes.

After several experiments and collecting data, I found proper parameter and data augmentation tricks and the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 118-133) consisted of a convolution neural network with the following fully-connected layers.

Here is a visualization of NVIDIA architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]

![alt text][image4]

![alt text][image5]

Then I repeated this process on the curves of the road in several times.

To augment the data sat, I also flipped images and angles, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

And I augment data by using left and right images. I use correction value = 0.15 after several experiments. And I also took reverse direction images by driving counter clockwise.

![alt text][image8]

![alt text][image9]
![alt text][image10]

I also collected a few number of images from the second track.

![alt text][image11]
![alt text][image12]

After the collection process, I had 17513 number of data points. It looks small but I increased it six times by augmentation. I then preprocessed this data by normalization and cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 after several experiments. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is my visualization about traing loss and validation loss. validation loss is fluctuated in the middle.

![alt text][image0]
