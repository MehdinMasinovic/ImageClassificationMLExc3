# CNN part of exercise 3

We chose the image classification task for our Machine Learning exercise. Namely, we chose to work with the Fashion MNIST dataset and the Labeled Faces in the Wild (LFW) people dataset. This part is related to the Convolutional Neural Network (CNN). The task of the exercise is to compare traditional methods with deep methods (CNN in this case) for classifying images. We used Python Notebooks here. This document describes the workflow and results.

## Datasets

The first task was to load the datasets into memory. We loaded the Fashion MNIST dataset as provided by tensorflow.keras.datasets, using the fashion_mnist function. The LFW people dataset was loaded by using the fetch_lfw_people function provided by sklearn.datasets. For the latter, we used code based on https://github.com/emanuelfakh/Face-Recognition/blob/master/FR_Final.ipynb. 

This notebook loaded only faces with at least 20 available examples. This rationale made sense to use: We want a minimum amount of faces per person to allow the network to have a minimum amount of faces to learn the facial features from. The original code extracted images sized 154x154 pixels with the three color channels red, green, and blue. The dimensions quickly filled up our GPU memory, so we adapted the code to scale the images down to 48x48 pixels. The number of examples were 3023, which were split up into 67% training and 33% test set.

The Fashion MNIST dataset consisted of greyscale images of 28x28 pixels with 60000 training and 10000 test examples.

## CNN architectures

We used CNN architectures we found on github under https://github.com/agoila/lisa-faster-R-CNN/tree/master/pyimagesearch/nn/conv. The priority was to use rather simple architectures to go easy on our resources time and memory.

The first architecture we used was MiniVGGNet which is a lightweight version of the VGGNet architecture. This architecture follows given requirements found to work especially well for images.

The second architecture we used was MiniGoogLeNet. It's based on GoogLeNet which is more complex than CNN. MiniGoogLeNet of it has a reduced set of parameters compared to GoogLeNet.

## Training

We started off training with MiniVGGNet and found, the more you trained, the better the validation accuracy, and the lower the loss, with deminishing returns. It would take a long time to converge. Thus, we decided to go with the results we would be getting from each of the setups after roughly 15 minutes of training on our machine.

Under this setting, learning_rate was a crucial hyperparameter. For each of the scenarios we tried learning rates 1e-4, 1e-3, 1e-2, and 1e-1 and then went with the setup yielding the best results after our time window. The batch size was kept at 32 for all of our setups.

We applied image augmentation for our setups and compared them to only using the original images when training. Viewing the images found in the LFW faces, it's possible that the faces are tilted or looking to the side a little. Image augmentation made sense to simulate such distortions on the training data. We sticked with the settings that Thomas Lidy used in the notebook mentioned in the exercise descriptions. Namely, the augmentations included rotations, width shifts, height shifts, zooming, and flipping.

The  Fashion MNIST photographs on the other hand were found to be highly standardized. Applying augmentation was expected to have little additional use. However we did go ahead and test with a limited set of augmentation consisting of rotations and flipping.

Later on, I repeated the same steps for MiniGoogLeNet and ran into the challenge that the original image sizes provided by the Fashion MNIST dataset (28x28 pixels) was too small for the architecture to work with. As the LFW people dataset had 48x48 pixels, we initially attempted to upscale the MNIST images to these dimensions also. However this was too much for our GPU. We found a spot that worked with 36x36 pixels so we sticked with that. Aside from that, for MiniGoogLeNet, we repeated the same steps as for MiniVGGNet.

To summarize, the following setups were tested:

- MiniVGGNet
  - Fashion MNIST
    - without augmentation
      - learning rate 1e-4
      - learning rate 1e-3
      - learning rate 1e-2
      - learning rate 1e-1
    - with augmentation
      - learning rate 1e-4
      - learning rate 1e-3
      - learning rate 1e-2
      - learning rate 1e-1
  - LFW Faces
    - without augmentation
      - learning rate 1e-4
      - learning rate 1e-3
      - learning rate 1e-2
      - learning rate 1e-1
    - with augmentation
      - learning rate 1e-4
      - learning rate 1e-3
      - learning rate 1e-2
      - learning rate 1e-1
- MiniGoogLeNet
  - Fashion MNIST
    - without augmentation
      - learning rate 1e-4
      - learning rate 1e-3
      - learning rate 1e-2
      - learning rate 1e-1
    - with augmentation
      - learning rate 1e-4
      - learning rate 1e-3
      - learning rate 1e-2
      - learning rate 1e-1
  - LFW Faces
    - without augmentation
      - learning rate 1e-4
      - learning rate 1e-3
      - learning rate 1e-2
      - learning rate 1e-1
    - with augmentation
      - learning rate 1e-4
      - learning rate 1e-3
      - learning rate 1e-2
      - learning rate 1e-1
      
## Findings and Results

MiniVGGNet without augmentations with a learning rate of 1e-2 yielded the best scores for the Fashion MNIST dataset. The dataset is balanced and the accuracy, precision, recall and f1 scores all yield a score of 0.92.

The LFW faces dataset was found to perform best with the MiniGoogLeNet architecture with augmentation enabled and a learning rate of 1e-1. The macro averages for precision: 0.87, recall: 0.83, and f1 score: 0.83. The weighted averages for precision: 0.90, recall: 0.88, f1 score: 0.88. The accuracy was 0.88.

It was found, however, that augmentation comes with a higher volatility when it comes to the learning curve for the validation set. With augmentation disabled, the learning rate 1e-1, the macro averages for precision: 0.84, recall: 0.82, and f1 score: 0.82. The weighted averages for precision: 0.87, recall: 0.86, f1 score: 0.86. The accuracy was 0.86.

The mistakes to be found in the confusion matrices for the MNIST dataset make sense: tops are most commonly mistaken as shirts. Pullovers are most commonly mistaken as coats and shirts. Shirts are most commonly mistaken as coats or tops. Ankle boots are most commonly mistaken as sneakers.

The confusion matrix for the LFW faces works best to illustrate the imbalance of the dataset: The most featured person in this dataset is George W. Bush, followed by Cole Powell and Donald Rumsfield. I cannot spot any people systematically being mistaken for one another using the confusion matrix.

We attempted to make the notebook reproducible, but the learning curves kept looking different regardless of all the seeds that we tried. We ended up just setting the numpy seed like Thomas Lidy did in his notebook (although it's still not 100% reproducible like this). Some sources suggested reproducibility by repetition and averaging. We have to agree with this. However continuously retraining the networks would exceed our time limits. We decided to go with one time limit. The learning curves to be found in the notebooks give an idea about the expected volatility of the results.
