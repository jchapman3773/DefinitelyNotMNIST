# Definitely Not MNIST

<img src="https://github.com/tdurnford/DefinitelyNotMNIST/blob/case-study/graphics/sample_letters.png"></img>

## Overview
Data consist of images of the letters A-J (10 classes) written in a variety of (unusual!) fonts. Image size is 28 x 28.

There were two datasets: a small dataset and a large dataset. The small dataset consisted of 18724 images while the large dataset had over half a million. We split the data into a training set and a validation set for both small and large sets of images.

## Convolutional Neural Network

### Structure

<img src="https://github.com/tdurnford/DefinitelyNotMNIST/blob/case-study/graphics/cnn.png"></img>


### Results

#### Small Dataset
Epoch 1/4
1310/1310 loss: 1.1496 - acc: 0.5975 - val_loss: 0.6673 - val_acc: 0.7483

Epoch 2/4
1310/1310 loss: 0.7729 - acc: 0.7274 - val_loss: 0.5691 - val_acc: 0.7816

Epoch 3/4
1310/1310 loss: 0.6777 - acc: 0.7535 - val_loss: 0.5587 - val_acc: 0.7805

Epoch 4/4
1310/1310 loss: 0.6548 - acc: 0.7592 - val_loss: 0.5466 - val_acc: 0.7805


#### Large Dataset

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

## Feature Extraction

<img src="https://github.com/tdurnford/DefinitelyNotMNIST/blob/julia/src/top_images.png"></img>

## I Had A Bad Idea -- Thanks Kelly

<img src="https://github.com/tdurnford/DefinitelyNotMNIST/blob/case-study/graphics/I had a bad idea.png"></img>

We decided to come up with a sentance and have our model read it... It ended up being a bad idea. 
The model we used was 

### Sentence Prediction
#### D DBE A GAD DDCA


## Future Work
