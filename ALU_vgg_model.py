from functions import image_gen_w_aug, model_output_for_TL, plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
import os
import datetime

train_dir = os.path.join('datasets/train/')
test_dir = os.path.join('datasets/test/')
val_dir = os.path.join('datasets/val/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, val_dir, test_dir)

pre_trained_model = tf.keras.applications.vgg16.VGG16(input_shape=(75, 75, 3),
                                                      include_top=False,
                                                      weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output

model_TL = model_output_for_TL(pre_trained_model, last_output)
model_TL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

log_dir = "logs/vgg/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model_TL.fit(
    train_generator,
    steps_per_epoch=10,  # 10 batches to train
    epochs=5,
    verbose=1,
    validation_data=validation_generator,
    callbacks=[tensorboard_callback])
