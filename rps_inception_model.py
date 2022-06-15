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
import datetime


def image_gen_w_aug(train_parent_directory, valid_test_parent_directory, test_parent_directory):
    train_datagen = ImageDataGenerator(rescale=1 / 255,
                                       rotation_range=30,
                                       zoom_range=0.2,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)

    test_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                        target_size=(75, 75),
                                                        batch_size=37,
                                                        class_mode='categorical')

    val_generator = train_datagen.flow_from_directory(valid_test_parent_directory,
                                                      target_size=(75, 75),
                                                      batch_size=37,
                                                      class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                      target_size=(75, 75),
                                                      batch_size=37,
                                                      class_mode='categorical')

    return train_generator, val_generator, test_generator


def model_output_for_TL(pre_trained_model, last_output):
    x = Flatten()(last_output)

    # Dense hidden layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Output neuron. 
    x = Dense(3, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)

    return model


train_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/train/')
test_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/test/')
val_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/val/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, val_dir, test_dir)

pre_trained_model = InceptionV3(input_shape=(75, 75, 3),
                                include_top=False,
                                weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed3')
last_output = last_layer.output

model_TL = model_output_for_TL(pre_trained_model, last_output)
model_TL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

log_dir = "logs/inception/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history_TL = model_TL.fit(
    train_generator,
    steps_per_epoch=10, #10 batches to train
    epochs=5,
    verbose=1,
    validation_data=validation_generator,
    callbacks=[tensorboard_callback])

tf.keras.models.save_model(model_TL, 'my_model.hdf5')
