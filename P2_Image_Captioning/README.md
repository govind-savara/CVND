[![Udacity Computer Vision Nanodegree](../images/cvnd.svg)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)

[image1]: ./imgs/image-captioning.png "Image Captioning"
[image2]: ./images/encoder-decoder.png "CNN Encoder and RNN Decoder"
[image3]: ./imgs/correct_image_captioning.png "Images with relatively accurate captions"
[image4]: ./imgs/incorrect_image_captioning.png "Images with relatively inaccurate captions"

# Image Captioning

## Project Overview
In this project, we will create a neural network architecture to automatically generate captions from images. 
After using the Microsoft Common Objects in COntext [(MS COCO) dataset](http://cocodataset.org/#home) to train our network, we will test our network on novel images!

![Image Captioning][image1] 

## Project Instructions
The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order:
- **0_Dataset.ipynb**       : Loading and Visualizing the MS COCO Data
- **1_Preliminaries.ipynb** : Experiment with CNN Encoder and Implement the RNN Decoder
- **2_Training.ipynb**      : Training the model with the MS COCO Data
- **3_Inference.ipynb**     : Inference on test data to auto generate image captions.

![CNN Encoder and RNN Decoder][image2]

### LSTM Decoder
In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: first a feature vector that is extracted from an input image, then a start word, then the next word, the next word, and so on! 

#### Embedding Dimension
The LSTM is defined such that, as it sequentially looks at inputs, it expects that each individual input in a sequence is of a **consistent size** and so we *embed* the feature vector and each word so that they are `embed_size`.

#### Sequential Inputs
So, an LSTM looks at inputs sequentially. In PyTorch, there are two ways to do this.  

The first is pretty intuitive: for all the inputs in a sequence, which in this case would be a feature from an image, a start word, the next word, the next word, and so on (until the end of a sequence/batch), you loop through each input like so:
```
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
```

The second approach, which this project uses, is to give the LSTM our entire sequence and have it produce a set of outputs and the last hidden state:

```
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state

# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
```

## Project Inference

Images with relatively accurate captions:

![image3]

Images with relatively inaccurate captions:

![image4]

**Note:**  A completely trained model is expected to take between 5-12 hours to train well on a GPU.