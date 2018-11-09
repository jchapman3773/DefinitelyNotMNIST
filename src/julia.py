from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.models import Model
import keras
import os
import numpy as np

# ## Delete corrupted files
# for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
#         directory = f'../data/{letter}/'
#         files = os.listdir(directory)
#         label = np.array([0]*10)
#         label[i] = 1
#         for file in files:
#             try:
#                 im = Image.open(directory+file)
#             except:
#                 print("Delete a corrupted file: " + file)
#                 os.remove(directory+file)
#                 continue

batch_size = 500
epochs = 5

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '../data/train',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '../data/validation',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='categorical')

def add_model_head(base_model, n_categories):
    """
    Takes a base model and adds a pooling and a softmax output based on the number of categories

    Args:
        base_model (keras Sequential model): model to attach head to
        n_categories (int): number of classification categories

    Returns:
        keras Sequential model: model with new head
        """

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_categories, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = ResNet50(weights='imagenet',include_top=False, input_shape=(32,32,3))
model = add_model_head(model, 10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
for layer in model.layers[:175]:
    layer.trainable = False
for layer in model.layers[175:]:
    layer.trainable = True

tensorboard = keras.callbacks.TensorBoard(
            log_dir='NotMNIST/', histogram_freq=0, batch_size=batch_size, write_graph=True, embeddings_freq=0)

mc = keras.callbacks.ModelCheckpoint('NotMNIST',
                                             monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.fit_generator(train_generator,steps_per_epoch=len(train_generator),
                                      epochs=epochs,
                                      validation_data=validation_generator,
                                      validation_steps=len(validation_generator),
                                      callbacks=[mc,tensorboard])

preds = model.predict_generator(validation_generator,500,callbacks=[mc,tensorboard])

score = model.evaluate_generator(validation_generator,500,callbacks=[mc,tensorboard])
print('Test score:', score[0])
print('Test accuracy:', score[1])

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
