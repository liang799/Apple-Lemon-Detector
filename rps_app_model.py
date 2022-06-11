# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 04:09:59 2020

@author: ASUS
"""

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os


def image_gen_w_aug(train_parent_directory, test_parent_directory):
    train_datagen = ImageDataGenerator(rescale=1 / 255,
                                       rotation_range=30,
                                       zoom_range=0.2,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       validation_split=0.15)

    test_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                        target_size=(75, 75),
                                                        batch_size=214,
                                                        class_mode='categorical',
                                                        subset='training')

    val_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                      target_size=(75, 75),
                                                      batch_size=37,
                                                      class_mode='categorical',
                                                      subset='validation')

    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                      target_size=(75, 75),
                                                      batch_size=37,
                                                      class_mode='categorical')

    return train_generator, val_generator, test_generator



train_dir = os.path.join('C:/Python/rps/datasets/train/')
test_dir = os.path.join('C:/Python/rps/datasets/test/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir)

model = tf.keras.models.Sequential([
  # Note the input shape is the desired size of the image:
  # 150x150 with 3 bytes color
  # This is the first convolution
  tf.keras.layers.Conv2D(64, (3,3), activation='relu',
              input_shape=(75, 75, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  # The second convolution
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # The third convolution
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # The fourth convolution
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # Flatten the results to feed into a DNN
  tf.keras.layers.Flatten(),
  # 512 neuron hidden layer
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=5,
    verbose=1,
    validation_data=validation_generator)

tf.keras.models.save_model(model, 'my_model.hdf5')
