# **Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**


[//]: # (Image References)

[image1]: ./output_images/whiteCarLaneSwitch.jpg "Predicted lane"

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

If you'd like to include images to show how the pipeline works, here is how to include an image: 

[image2]: ./output_images/solidYellowCurve2.jpg "Original Image"

[image3]: ./output_images/gray_image.jpg "Grayscale"

[image4]: ./output_images/canny_image.jpg "Canny transform"

[image5]: ./output_images/blurred_imgae.jpg "Gaussian blur"

[image6]: ./output_images/interesting_image.jpg "Region of interest"

[image7]: ./output_images/lined_image.jpg "Segments detection with hough transform"

[image8]: ./output_images/solidYellowCurve2.jpg "Extrapolation"


### 2. Potential shortcomings with the current pipeline

Some shortcomings of the current method :
1. The case when the line is curvy would not be detected
2. The case when the car is not aligned with the lane, while overtaking another car for example

### 3. Possible improvements

Some possible improvements of the current method :
1. For the first shortcoming, extrapolating between every segment
2. Using the information that the car is in the process of overtaking, and using other parameters tuned for this specific case

