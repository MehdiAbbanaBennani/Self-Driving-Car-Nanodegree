#**Traffic Sign Recognition**


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

![image1](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/master/Traffic-Sign-Classifier/german_traffic_signs/1.png)
![image2](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/master/Traffic-Sign-Classifier/german_traffic_signs/2.png)
![image3](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/master/Traffic-Sign-Classifier/german_traffic_signs/3.png)
![image4](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/master/Traffic-Sign-Classifier/german_traffic_signs/4.png)
![image5](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/master/Traffic-Sign-Classifier/german_traffic_signs/5.png)

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/MehdiAB161/Self-Driving-Car-Nanodegree/blob/master/Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the label counts. The dataset is unbalanced, which might biais the prediction.

![Train set labels distribution](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/Behavioral-Cloning/Traffic-Sign-Classifier/doc_images/train-labels-distribution.png)

The following figure shows the distribution of the pixels for the initial train set :

![Train set pixels distribution](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/Behavioral-Cloning/Traffic-Sign-Classifier/doc_images/train-pixel-distribution.png)

###Design and Test a Model Architecture

####1. Image Preprocessing

As a first step, I decided to generate additional images using rotations and noise blurring, in order to rebalance the train set. Later, I realized that the increase in the performance was not significant and I considered only the initial data.

As a last step, I normalized the image data because the learning speed increases if the variance of each coordinate of the input vector is equal to 1.

The following figure shows the distribution of the pixels for the normalized train set.

![Normalized train set pixels distribution](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/Behavioral-Cloning/Traffic-Sign-Classifier/doc_images/train-normaliszed-pixels-distribution.png)


####2. Model description
My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Flatten   	      	| outputs 800x1 								|
| Dropout   	      	| 0.05 dropout probability						|
| Fully connected		| 200 nodes,   outputs 200x1					|
| Fully connected		| 98 nodes,   outputs 98x1  					|
| Fully connected		| 43 nodes,   outputs 43x1						|
| Softmax				|												|

It is very similar to the LeNet neural network. The differences are :
  * An additional dropout layer after flattening the convolutional layers output
  * Larger fully connected layers in order to capture more features, because the number of possibles classes is higher (43 instead of 10).

####3. Model training

To train the model, I used the Adam Optimizer. The model ran 50 epochs.

####4. Model selection
I considered several neural networks architectures, mostly inspired from LeNet architecture. I tried adding several dropout layers, adding an additional convolutional layer, changing the size of the layers.

I also considered learning rate decay, and other optimization algorithms. I also tested several sets of parameters.

Dropout helps achieving a better generalization performance.

Since all the pictures represent traffic signs, the class of necessary features for the classification is smaller than the needed class for more complex problems which involve very different object. Therefore, I think a shallow neural network is enough for the discrimination. However, since there are 43 classes, the classification quality improved when I considered wider fully connected layers in comparison with the LeNet layers. The reason is that LeNet was designed for MNIST which only contains 10 classes.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.95
* test set accuracy of 0.94

* The test accuracy is 94%, since the data is unseen, it proves that the model works well. However, the train accuracy of 1 suggest that the model is overfitting to the train set.

###Test a Model on New Images

####1. Five German traffic signs images from the web
Here are five German traffic signs that I found on the web:

![image1](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/master/Traffic-Sign-Classifier/german_traffic_signs/1.png)
![image2](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/master/Traffic-Sign-Classifier/german_traffic_signs/2.png)
![image3](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/master/Traffic-Sign-Classifier/german_traffic_signs/3.png)
![image4](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/master/Traffic-Sign-Classifier/german_traffic_signs/4.png)
![image5](https://raw.githubusercontent.com/MehdiAB161/Self-Driving-Car-Nanodegree/master/Traffic-Sign-Classifier/german_traffic_signs/5.png)

The images all centered and clear, except the images 4 and 5, which are noisy. The difficulty might come from the shapes around the images, which could impact the classifier.

####2. Discussing the prediction

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No entry      		| No entry   									|
| Go straight  			| Keep right 									|
| Turn left				| Stop											|
| Speed limit (60km/h)	| Speed limit (60km/h)			 				|
| Yield     			| Yield      						        	|


The model was able to correctly guess 3 of the 5 traffic signs, which corresponds to an accuracy of 60%. On the other hand, the accuracy over the test set was 0.95, we expected a prediction accuracy of 0.8 or 1. But since the sample of new images is very small, it is a very noisy estimate.

The contrast of the sign "Turn left" is very similar to "Stop", which could have missled the classifier.

####3. Top 5 softmax probabilities


For the first image, the model is sure that this is a No entry sign (probability of 1), and the image does contain a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| No entry   									|
| .00     				| Turn right ahead  							|
| .00					| Turn left ahead								|
| .00	      			| No passing					 				|
| .00				    | Beware of ice/snow         					|


For the second image, the model is sure that this is a Keep right sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Keep right   									|
| .00     				| End of all speed and passing limits			|
| .00					| Turn left ahead								|
| .00	      			| Stop					 			        	|
| .00				    | Priority road      							|

For the third image, the model is relatively sure that this is a Stop sign (probability of 0.77), and the image does contain a Turn left sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .77         			| Stop   									    |
| .22     				| Right-of-way at the next intersection 		|
| .01					| Traffic signals								|
| .00	      			| Priority road					 				|
| .00				    | Speed limit (80km/h)      					|

For the fourth image, the model is sure that this is a Speed limit (60km/h) sign (probability of 1), and the image does contain a Speed limit (60km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Speed limit (60km/h)   						|
| .00     				| Speed limit (50km/h) 							|
| .00					| Speed limit (80km/h)							|
| .00	      			| No vehicles					 				|
| .00				    | End of all speed and passing limits			|

For the fifth image, the model is sure that this is a Yield sign (probability of 1), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Yield   									    |
| .00     				| Turn left ahead 							    |
| .00					| Ahead only									|
| .00	      			| Priority road					 				|
| .00				    | Keep right      							    |
