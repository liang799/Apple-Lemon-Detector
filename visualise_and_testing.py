import tensorflow as tf
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.utils import img_to_array
from matplotlib import pyplot
from numpy import expand_dims
from functions import image_gen_w_aug, plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

train_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/train/')
val_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/val/')
test_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/test/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, val_dir, test_dir)

model = tf.keras.models.load_model('C:/Python/Apple-Lemon-Detector/my_model.hdf5')  # loading a trained model
results1 = model.evaluate(test_generator, batch_size=None, verbose=2)
print("test loss, test acc:", results1)

model = Model(inputs=model.inputs, outputs=model.layers[1].output)

# Visualising Image augmentation
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = train_generator.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()

# Visualising Feature Maps
img, label = test_generator.next()
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
    for _ in range(square):
        # specify subplot and turn of axis
        ax = pyplot.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
        ix += 1
# show the figure
pyplot.show()
