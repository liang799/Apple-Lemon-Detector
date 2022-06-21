import tensorflow as tf
from keras.models import Model
from matplotlib import pyplot
from functions import image_gen_w_aug
import os

train_dir = os.path.join('datasets/train/')
val_dir = os.path.join('datasets/val/')
test_dir = os.path.join('datasets/test/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, val_dir, test_dir)

model = tf.keras.models.load_model('models/tune.hdf5')  # loading a trained model
results1 = model.evaluate(test_generator, batch_size=None, verbose=2)
print("test loss, test acc:", results1)

model = Model(inputs=model.inputs, outputs=model.layers[1].output)

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
