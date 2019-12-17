## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        # output size = (W-F+2p)/S + 1 = (224-5+4)/1 + 1 = 224
        # output tensor dimension: (32, 224, 224)
        self.pool1 = nn.MaxPool2d(4, 4)         # output size = (224-4)/4 + 1 = 56
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)       # output size = (56-3+2)/1 + 1 = 56
        self.pool2 = nn.MaxPool2d(2, 2)         # output size = (56-2)/2 + 1 = 28
        self.conv3 = nn.Conv2d(64, 128, 1)      # output size = (28-1)/1 + 1 = 28
        self.pool3 = nn.MaxPool2d(2, 2)         # output size = (28-2)/2 + 1 = 14
        self.conv4 = nn.Conv2d(128, 256, 1)     # output size = (14-1)/1 + 1 = 14
        self.pool4 = nn.MaxPool2d(2, 2)         # output size = (14-2)/2 + 1 = 7

        self.fc1 = nn.Linear(256 * 7 * 7, 1024)  # 256 outputs and 7*7 pooled map
        self.fc2 = nn.Linear(1024, 136)          # 136 output channels, 2 for each 68 keypoint (x, y) pairs

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)



    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.relu(self.conv3(x))))
        x = self.dropout4(self.pool4(F.relu(self.conv4(x))))

        # flattening the feature map
        x = x.view(x.size(0), -1)

        # 3 linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)
        # x = self.dropout2(x)
        # x = F.relu(self.fc3(x))

        # a modified x, having gone through all the layers of your model, should be returned
        return x
