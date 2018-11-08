# Definitely Not MNIST

<img src="https://github.com/tdurnford/DefinitelyNotMNIST/blob/case-study/graphics/sample_letters.png"></img>
## The Squad
Millie, Julia, TJ

## Overview
Data consist of images of the letters A-J (10 classes) written in a variety of (unusual!) fonts. Image size is 28 x 28.

There were two datasets: a small dataset and a large dataset. The small dataset consisted of 18724 images while the large dataset had over half a million. We split the data into a training set and a validation set for both small and large sets of images.

## Convolutional Neural Network

### Structure

<img src="https://github.com/tdurnford/DefinitelyNotMNIST/blob/case-study/graphics/cnn.png"></img>


### Results

#### Small Dataset
Epoch 1/4
13106/13106 loss: 0.5870 - acc: 0.7774 - val_loss: 0.4741 - val_acc: 0.8170

Epoch 2/4
13106/13106 loss: 0.4193 - acc: 0.8321 - val_loss: 0.5086 - val_acc: 0.8229


#### Large Dataset

Epoch 1/4
9232/9232 loss: 0.8156 - acc: 0.7119 - val_loss: 0.6174 - val_acc: 0.7642

Epoch 2/4
9232/9232 loss: 0.6821 - acc: 0.7570 - val_loss: 0.5752 - val_acc: 0.7912

Epoch 3/4
9232/9232 loss: 0.6440 - acc: 0.7741 - val_loss: 0.5599 - val_acc: 0.7954

Epoch 4/4
9232/9232 loss: 0.6175 - acc: 0.7868 - val_loss: 0.5319 - val_acc: 0.8091

### Random Forest

Once we were done tuning the CNN, we took the output from the Flatten layer and fed the data into a Random Forest. However, the accuarcy score was lower than our initial accuarcy with the CNN. 

## Transfer Learning

### Structure
The Xception model was used as the base model. The model head was replaced and fine tuning of the model on our data was attempted. Softmax was used for the activation function for predictions. 

The keras ImageDataGenerator was used as a generator for train and test data with Xception's preprocessing function. 
Zoom and shear were applied to the training data.

The model head was initially trained over 5 epochs and then additional layers were unlocked for training over 10 additional epochs.

### Results
Training accuracy and loss (categorical cross entropy) improved from Epoch to epoch. However, loss for the validation set was unstable.

Epoch 1/10
265/265 loss: 1.0246 - acc: 0.7254 - val_loss: 5.7956 - val_acc: 0.4986

Epoch 2/10
265/265 loss: 0.5908 - acc: 0.8773 - val_loss: 0.9881 - val_acc: 0.7433

Epoch 3/10
265/265 loss: 0.5228 - acc: 0.8952 - val_loss: 1.5174 - val_acc: 0.7568

...

Epoch 6/10
265/265 loss: 0.4463 - acc: 0.9153 - val_loss: 0.6321 - val_acc: 0.8407

Epoch 7/10
265/265 loss: 0.3779 - acc: 0.9255 - val_loss: 1.6321 - val_acc: 0.7480


## Feature Extraction

Used ResNet50 trained on ImageNet predictions on DefinitelyNotMNIST images into NMF with 10 components.

<img src="https://github.com/tdurnford/DefinitelyNotMNIST/blob/julia/src/top_images_10.png"></img>

## I Had A Bad Idea -- Thanks Kelly

<img src="https://github.com/tdurnford/DefinitelyNotMNIST/blob/case-study/graphics/I had a bad idea.png" width="100%"></img>

We decided to come up with a sentance and have our model read it... It ended up being a bad idea. 
The model we used was based on the small dataset.. maybe it'll work better wit the large dataset. 

### Sentence Prediction
#### D DBE A GAD DDCA


