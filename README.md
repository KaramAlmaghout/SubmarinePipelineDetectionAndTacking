## SubmarinePipelineDetectionAndTacking

Pipeline detection and tracking approaches for autonomous underwater vehicler


### pre-requirements

OpenCV library >= 3.2.*
Eigen library


### pipline detection files:

### compile

g++ <file_name>.cpp -o <file_name>  `pkg-config --cflags --libs opencv`

### run

./<file_name> 

or

imgs/<img_name>


### Class alphaFilter is a Kalman filter to estimate the pipeline position during the tracking 
Input: 
*Z: detected position*
*delta_x: difference between position at sample K and sample K-1*
Output:
*X_m: estimated position*

