## SubmarinePipelineDetectionAndTacking

Pipeline detection and tracking approaches for autonomous underwater vehicle


## pre-requirements

OpenCV library >= 3.2.*

Eigen library


## pipline detection files:

### compile

g++ <file_name>.cpp -o <file_name>  `pkg-config --cflags --libs opencv`

### run

./<file_name> 

or

imgs/<img_name>

## Results

### pipelineIntersectLines

![image](https://user-images.githubusercontent.com/94979970/189586375-6b8745e2-19e9-4d7b-a1eb-ad85b11cad86.png)

![image](https://user-images.githubusercontent.com/94979970/189586422-fb0211bf-f762-484c-a5cf-28bfece9e874.png)

![image](https://user-images.githubusercontent.com/94979970/189586446-cbe79f57-f0c4-429c-8619-891ca98a09f2.png)

### pipelineSOM

![image](https://user-images.githubusercontent.com/94979970/189586515-58fa8520-3a70-4521-9049-6216e83d0b13.png)

![image](https://user-images.githubusercontent.com/94979970/189586542-26850301-ee42-4f2e-8d50-6f20b0ff592b.png)

![image](https://user-images.githubusercontent.com/94979970/189586569-ce3abfed-1913-4599-ba90-16d46aaa764b.png)



### Class alphaFilter is a Kalman filter to estimate the pipeline position during the tracking 
Input: 

*Z: detected position*

*delta_x: difference between position at sample K and sample K-1*

Output:

*X_m: estimated position*

