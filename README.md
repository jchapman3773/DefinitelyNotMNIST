# Definitely Not MNIST

<img src="https://github.com/tdurnford/DefinitelyNotMNIST/blob/case-study/graphics/sample_letters.png"></img>

## Overview
Data consist of images of the letters A-J written in a variety of (unusual!) fonts. Image size is 28 x 28.

There were two datasets: a small dataset and a large dataset. The small dataset consisted of 18724 images while the large dataset had 
The data was split into training (13201 images) and validation (5523 images) sets.

## Convolutional Neural Network

### Structure
We started by building a Convolutional Nueral Network to approach this problem. Our final model 
### Results

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

## Ransom Letter Translation 

## Future Work
