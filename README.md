# Definitely Not MNIST

<img src="https://github.com/tdurnford/DefinitelyNotMNIST/blob/case-study/graphics/sample_letters.png"></img>

## Overview
Data consist of images of the letters A-J written in a variety of (unusual!) fonts. Image size is 28 x 28.

The data was split into training (13201 images) and validation (5523 images) sets.

## Convolutional Neural Network

### Structure

### Results

## Transfer Learning

### Structure
The Xception model was used as the base model. The model head was replaced and fine tuning of the model on our data was attempted. Softmax was used for the activation function for predictions. 

The keras ImageDataGenerator was used as a generator for train and test data with Xception's preprocessing function. 
Zoom and shear were applied to the training data.

The model head was initially trained over 5 epochs and then additional layers were unlocked for training over 10 additional epochs.

### Results

## I Had a Bad Idea -- Thanks Kelly

<img src="https://github.com/tdurnford/DefinitelyNotMNIST/blob/case-study/graphics/I had a bad idea.png"></img>

We decided to predict each letter in our well crafted sentance using our Convolutional Nueral Network trained on our small dataset.. 

### D DBE A GAD DDCA

## Future Work
