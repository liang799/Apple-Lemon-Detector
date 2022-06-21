from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image

image_path = 'datasets//train//lemon//1 (212).jpg'

# Visualise original image
im = Image.open(image_path)
plt.imshow(im)
plt.show()

# Loads image in from the set image path
img = keras.preprocessing.image.load_img(image_path, target_size=(75, 75))
img_tensor = keras.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Uses ImageDataGenerator to flip the images
datagen = ImageDataGenerator(# rescale=1. / 255,
                             rotation_range=30,
                             zoom_range=0.2,
                             width_shift_range=0.1,
                             height_shift_range=0.1)
# Creates our batch of one image
pic = datagen.flow(img_tensor, batch_size=1)
plt.figure(figsize=(16, 16))
# Plots our figures
for i in range(1, 17):
    plt.subplot(4, 4, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_)
plt.show()
