[![Udacity Computer Vision Nanodegree](../images/cvnd.svg)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)

# Landmark Detection & Robot Tracking (SLAM)

## Project Overview
In this project, implemented SLAM (Simultaneous Localization and Mapping) for a 2 dimensional world! We will combine what we know about robot sensor measurements and movement to create a map of an environment from only sensor and motion data gathered by a robot, over time.
SLAM gives us a way to track the location of a robot in the world in real-time and identify the locations of landmarks such as buildings, trees, rocks, and other world features. 
This is an active area of research in the fields of robotics and autonomous systems. 

Below is an example of a 2D robot world with landmarks (purple x's) and the robot (a red 'o') located and found using only sensor and motion data collected by that robot. 
This is just one example for a 50x50 grid world; in our project work we will likely generate a variety of these maps.

![Example of SLAM output (estimated final robot pose and landmark locations)](./images/robot_world.png)

## Project Instructions
The project will be broken up into three Python notebooks; the first two are for exploration of provided code, and a review of SLAM architectures, only Notebook 3 and the robot_class.py file will be graded:

- **Notebook 1** : Robot Moving and Sensing
- **Notebook 2** : Omega and Xi, Constraints 
- **Notebook 3** : Landmark Detection and Tracking 

Follow the notebooks to more instructions on the project.