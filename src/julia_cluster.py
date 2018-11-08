from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.models import Model
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import os
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({
    'figure.figsize'      : (15,15),
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
    'legend.loc'          : 'upper right'
})

# batch_size = 500
# epochs = 5
#
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2)
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#         '../data/train',
#         target_size=(224, 224),
#         batch_size=batch_size,
#         class_mode='categorical')
#
# validation_generator = test_datagen.flow_from_directory(
#         '../data/validation',
#         target_size=(224, 224),
#         batch_size=batch_size,
#         class_mode='categorical')

model = ResNet50(weights='imagenet')

resnet_feature_list = []
file_paths = []

for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
    directory = f'../data/train/{letter}/'
    files = os.listdir(directory)
    label = np.array([0]*10)
    label[i] = 1
    for idx,file in enumerate(files[:10]):
        img = image.load_img(directory+file, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        resnet_feature = model.predict(img_data)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())
        file_paths.append(directory+file)
        # if idx%1 == 0:
            # print(f'{idx}: {directory+file}')
        print(directory+file)

resnet_feature_list_np = np.array(resnet_feature_list)
nmf_model = NMF(n_components=10, random_state=42).fit(resnet_feature_list_np)
nmf_w = nmf_model.transform(resnet_feature_list_np)
nmf_H = nmf_model.n_components_

top_images = 10
feat_idx = []
for class_num in range(10):
    feat_idx.append(np.argsort(nmf_w[:,class_num])[::-1][:top_images])
fig,axes = plt.subplots(10,top_images)
for idx1,row in enumerate(feat_idx):
    for idx2,img_idx in enumerate(row):
        axes[idx1][idx2].imshow(mpimg.imread(file_paths[img_idx]))
        # axes[idx1][idx2].tick_params(axis='x',labelbottom='off')
        # axes[idx1][idx2].tick_params(axis='y',labelbottom='off')
        axes[idx1][idx2].get_xaxis().set_visible(False)
        axes[idx1][idx2].get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig('top_images_10.png')
plt.show()

# def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
#     plt.figure(figsize=(2. * n_col, 2.26 * n_row))
#     plt.suptitle(title, size=16)
#     for i, comp in enumerate(images):
#         plt.subplot(n_row, n_col, i + 1)
#         vmax = max(comp.max(), -comp.min())
#         plt.imshow(comp.reshape(image_shape), cmap=cmap,
#                    interpolation='nearest',
#                    vmin=-vmax, vmax=vmax)
#         plt.xticks(())
#         plt.yticks(())
#     plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
# plot_gallery("First ten images", faces_centered[:n_components])
