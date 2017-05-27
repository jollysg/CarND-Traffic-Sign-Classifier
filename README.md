#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./resources/visualization.png "Visualization"
[image2]: ./resources/grayscaling.png "Grayscaling"
[image3]: ./resources/augmentation.png "Augmentation"
[image4]: ./resources/augmentedDatasetVisualization.png "Augmented dataset"
[image5]: ./resources/trafficSignsFromWeb.png "Traffic Signs from web for testing"
[image6]: ./resources/predictedTrafficSigns.png "Predictions for the traffic signs"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jollysg/CarND-Traffic-Sign-Classifier)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 50.98 by 50.45 on an average
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the training and testing data for every class. Numpy's bincount method was used to obtain the frequncy distribution of the data for every class.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces the channels of the image, so that a CNN with lower depth channels can be used. The conversion was done using color.rgb2gray method, which also took care of normalization as it returns the gray images with pixel ranges from 0 to 1 float. As a result, a separate normalization step wasn't needed. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

These images were then used to train a CNN which had the LeNet's architecture basically. The validation accuracy achieved with this was about 89%.
 
After that I decided to generate additional data because, as can be seen from the visualization of the dataset, there were quite a few classes for which the number of training samples were very less. To prevent the network to be biased towards the classes that had more samples, I decided to augment the images for classes with lower samples. 

To add more data to the the data set, I rotated the images randomly between -20 and +20 degrees. I calculated the mean number of samples of all the classes for the original data before augmentation. The goal of augmentation was to generate additional images so that all the classes had a certain minimum number of samples, given by a factor (in this case 1.8) of the mean bin size. 

Here is an example of an original image and an augmented image:

![alt text][image3]

The size of the augmented data training set was increased to 65875 samples. Here is a visualization showing the distribution of the augmented data for each class.

![alt text][image4]

However, this increased the validation accuracy of the network only marginally to about 90-90.5%. The reason for this seems to be the validation data set, which had a similar frequency distribution of the classes to the original training dataset.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	  				|
| Fully connected		| 400 inputs, 120 outputs        				|
| RELU					|												|
| Fully connected		| 120 inputs, 84 outputs        				|
| RELU					|												|
| Fully connected		| 84 inputs, 43 outputs (number of classes)     |
| Softmax				| 	        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer. I experimented with the learning rate initially. With the optimum learning rate of 0.0005 I was able to achieve 89% of accuracy initially. After that I augmented the data as described in the section above. That increased the validation accuracy to about 91%. I then implemented 50% dropouts for the first two fully connected layers while training. This increased the accuracy further to about 93%. Finally, I added L2 regularization. With the lambda of 0.0005, I was able to achieve an accuracy of about 96% for the network by training it for 40 epochs. The test data accuracy of 93% was achieved using this. As a future step, to increase the accuracy of the network further, additional fully connected and/or convolution layers can be added to the network. Another alternative can be to include all the color channels of the images as features.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
The previous section summarized the approach taken for achieving the goal of the validation accuracy. The network chosen in this case was LeNet as the image shapes in this dataset were similar to the LeNet dataset. It was also confirmed that CNNs function the same irrespective of the content of the images.  
My final model results were:
* training set accuracy of ~98%
* validation set accuracy of ~96% 
* test set accuracy of ~93%

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Following are the five German traffic signs that I found on the web. 
![alt text][image5]

![alt text][image6]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The images were resized to 32 x 32 in an external image editor, further the unnecessary regions of the images were cropped out. An accuracy of 80% was achieved with this. Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h 	    		| 30 km/h   									| 
| Right of way     		| Right of way 									|
| No passing			| No passing 									|
| Children crossing     | Children crossing				 				|
| Stop					| Stop			      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model can be seen in the final cells of the  Ipython notebook.

For the first image, the model confuses the 70 km/h speed limit sign with a 30 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .75         			| 30 km/h   									| 
| .24     				| 20 km/h										|
| .01					| 70 km/h										|
| .00	      			| Keep right					 				|
| .00				    | 50 km/h		      							|


For the second image - right of way at the next intersection: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Right of way at next intersection   			| 
| .00     				| Beware of ice/snow							|
| .00					| General caution								|
| .00	      			| Priority road					 				|
| .00				    | Pedestrians	      							|

For the third image - No passing: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No passing						  			| 
| .00     				| Dangerous curve to the left					|
| .00					| End of no passing								|
| .00	      			| Vehicles over 3.5 metric tons prohibited	 	|
| .00				    | No passing for vehicles over 3.5 metric...	|


For the fourth image - Children crossing: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Children crossing					  			| 
| .38     				| Right of way at the next intersection			|
| .01					| Dangerous curve to the right					|
| .01	      			| Pedestrians					 				|
| .00				    | Beware of ice/snow	    					|

For the fifth image - Stop: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop								  			| 
| .38     				| Turn right ahead								|
| .01					| Keep left										|
| .01	      			| Turn left ahead				 				|
| .00				    | No entry	    					|

