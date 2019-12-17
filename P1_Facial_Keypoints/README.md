[![Udacity Computer Vision Nanodegree](../images/cvnd.svg)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"
[image2]: ./imgs/model_result.JPG "Model Predictions"
[image3]: ./imgs/Face_and_Facial_keypoint_detection_result.jpg "Face and Facial Keypoint detection Result"

# Facial Keypoint Detection

In this project using the knowledge of computer vision techniques and deep learning architectures build a facial keypoint detection system. 
Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. 
These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. 
The project able to look at any image, detect faces, and predict the locations of facial keypoints on each face.

Some examples of these keypoints are pictured below.

![Facial Keypoint Detection][image1]

## Model Architecture

I tried using the NaimishNet with little modifications. The model architecture shown below:

|   Layer         		|     Specifications    	        			            | 
|:---------------------:|---------------------------------------------------------| 
|   Input |  size: (224, 224, 1)     |
|   Conv 1  |    filters: 32, Kernel size: (5, 5), padding: (2, 2), Activation: ReLU   |
|   Max Pooling |    kernel size: (4, 4), stride: (4, 4)    |
|   Dropout    |    probability: 0.1    |
|   Conv 2  |   filters: 64, kernel size: (3, 3), padding: (1, 1), Activation: ReLU |
|   Max Pooling |   kernel size: (2, 2), stride: (2, 2)    |
|   Dropout    |    probability: 0.2    |
|   Conv 3  |   filters: 128, kernel size: (1, 1), Activation: ReLU |
|   Max Pooling |   kernel size: (2, 2), stride: (2, 2)    |
|   Dropout    |    probability: 0.3    |
|   Conv 4  |   filters: 256, kernel size: (1, 1), Activation: ReLU |
|   Max Pooling |   kernel size: (2, 2), stride: (2, 2)    |
|   Dropout    |    probability: 0.4    |
|   Fully Connected  |   Input Channels: 12544, Output Channels: 1024, Activation: ReLU |
|   Dropout    |    probability: 0.5    |
|   Fully Connected  |   Input Channels: 1024, Output Channels: 136 |

## Result:

The model predictions with truth keypoints are shown below:

![Model Prediction Result][image2]

The full pipeline with face detection, extraction and keypoints prediction results are shown below:

![Face and Facial Keypoint detection result][image3]

__Note:__ Project instructions can be found in the [Udacity README](README_Udacity.md).