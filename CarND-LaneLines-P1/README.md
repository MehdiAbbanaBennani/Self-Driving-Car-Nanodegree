# **Finding Lane Lines on the Road**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  
[//]: # (Image References)


![Predicted lane](https://github.com/MehdiAB161/Self-Driving-Car-Nanodegree/blob/LaneP1/CarND-LaneLines-P1/output_images/solidYellowCurve.jpg?raw=true "Predicted lane")

<a href="https://youtu.be/yie5K0BM1fs
" target="_blank"><img src="http://img.youtube.com/vi/yie5K0BM1fs/0.jpg"
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>


### Reflection

### 1. Pipeline description

My pipeline consisted of 6 steps :
1. Convert the images to grayscale
2. Compute the Canny transform
3. Smoothing the image with Gaussian smoothing
4. Extracting the current line from the image using a mask
5. Detecting the lines in the region of interest using Hough transform
6. Extrapolating line segments with linear extrapolation

In order to draw a single line on the left and right lanes, I coded the full_lanes_with_average() function, which :
1. Extracts both sides of the lane
2. For each side, uses the average slope between the uppermost and lowest points to draw a line


![0: Original Image](https://github.com/MehdiAB161/Self-Driving-Car-Nanodegree/blob/LaneP1/CarND-LaneLines-P1/test_images/solidYellowCurve2.jpg?raw=true "0: Original Image")

![Grayscale](https://github.com/MehdiAB161/Self-Driving-Car-Nanodegree/blob/LaneP1/CarND-LaneLines-P1/output_images/gray_image.jpg?raw=true "Grayscale")

![Canny transform](https://github.com/MehdiAB161/Self-Driving-Car-Nanodegree/blob/LaneP1/CarND-LaneLines-P1/output_images/canny_image.jpg?raw=true "Canny transform")

![Gaussian blur](https://github.com/MehdiAB161/Self-Driving-Car-Nanodegree/blob/LaneP1/CarND-LaneLines-P1/output_images/blurred_image.jpg?raw=true "Gaussian blur")

![Region of interest](https://github.com/MehdiAB161/Self-Driving-Car-Nanodegree/blob/LaneP1/CarND-LaneLines-P1/output_images/interesting_image.jpg?raw=true "Region of interest")

![Segments detection with hough transform](https://github.com/MehdiAB161/Self-Driving-Car-Nanodegree/blob/LaneP1/CarND-LaneLines-P1/output_images/lined_image.jpg?raw=true "Segments detection with hough transform")

![Extrapolation](https://github.com/MehdiAB161/Self-Driving-Car-Nanodegree/blob/LaneP1/CarND-LaneLines-P1/output_images/solidYellowCurve.jpg?raw=true "Extrapolation")


### 2. Potential shortcomings with the current pipeline

Some shortcomings of the current method :
1. The case when the line is curvy would not be detected
2. The case when the car is not aligned with the lane, while overtaking another car for example

### 3. Possible improvements

Some possible improvements of the current method :
1. For the first shortcoming, extrapolating between every segment
2. Using the information that the car is in the process of overtaking, and using other parameters tuned for this specific case
