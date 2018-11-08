#!/usr/bin/env python3
import numpy as np
from keras import backend as K
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.ensemble import Log

# dimensions of our images.
img_width, img_height = 28, 28

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'

nb_train_samples = 13106
nb_validation_samples = 5620
epochs = 1
batch_size = 25

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=nb_train_samples,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=nb_validation_samples,
    class_mode='categorical')

model = load_model('notMNIST.h5')
features = Model(model.layers[0].input, model.layers[-6].output)

# print('predicting....')
# print(features.predict_generator(train_generator, steps=nb_train_samples).shape)

X_train, y_train = train_generator.next()
X_test, y_test = validation_generator.next()

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

X_train = features.predict(X_train)
X_test = features.predict(X_test)

rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))