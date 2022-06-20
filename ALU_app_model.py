import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
from keras.callbacks import EarlyStopping
from functions import image_gen_w_aug, plot_confusion_matrix
import os
import matplotlib.pyplot as plt

train_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/train/')
val_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/val/')
test_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/test/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, val_dir, test_dir)


def build_model(hp):
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(75, 75, 3)))
    for i in range(hp.Int("conv_layers", 3, 5, default=3)):
        model.add(
            layers.Conv2D(
                hp.Int(f"filters_{i}", min_value=32, max_value=96, step=32),
                kernel_size=(3, 3),
                padding="same"
            )
        )
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=-1))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))

    # Dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(hp.Int("dense_units", min_value=256,
                                  max_value=768, step=256)))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())

    # softmax classifier
    model.add(layers.Dense(3))
    model.add(layers.Activation("softmax"))

    # initialize the learning rate choices and optimizer
    lr = hp.Choice("learning_rate",
                   values=[1e-1, 1e-2, 1e-3])
    opt = keras.optimizers.Adam(learning_rate=lr)

    # compile the model
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search_space_summary()
tuner.search(train_generator,
             validation_data=validation_generator,
             epochs=30,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])