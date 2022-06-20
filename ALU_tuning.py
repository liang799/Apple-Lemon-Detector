import tensorflow as tf
import keras_tuner
import os
from tensorflow import keras
from keras import callbacks, regularizers
from tensorflow.keras import layers
from functions import image_gen_w_aug
import matplotlib.pyplot as plt

train_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/train/')
val_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/val/')
test_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/test/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, val_dir, test_dir)


def build_model(hp):
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(75, 75, 3)))

    # Conv layers
    for i in range(hp.Int("conv_layers", min_value=3, max_value=5, step=1)):
        model.add(
            layers.Conv2D(
                hp.Int(f"filters_{i}", min_value=32, max_value=128, step=32),
                kernel_size=(3, 3),
                padding="same"
            )
        )
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(hp.Float(f"dropout_{i}", 0, 0.5, step=0.1)))

    # Dense layer
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    punish = hp.Choice("regularizer", values=[1e-1, 1e-2, 1e-3, 1e-4])
    model.add(layers.Dense(hp.Int("dense_units", min_value=256, max_value=768, step=256),
                           kernel_regularizer=regularizers.l2(punish)))
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(hp.Float("dense_dropout", 0, 0.5, step=0.1)))

    # softmax classifier
    model.add(layers.Dense(3))
    model.add(layers.Activation("softmax"))

    # compile the model
    model.compile(optimizer='adam', loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


build_model(keras_tuner.HyperParameters())

earlystopping = callbacks.EarlyStopping(monitor="val_loss", patience=1)

tuner = keras_tuner.tuners.BayesianOptimization(
    hypermodel=build_model,
    objective='val_loss',
    max_trials=1,
    overwrite=True,
    directory="logs",
    project_name="bae-tuning-v2",
)
# tuner.search_space_summary()

tuner.search(train_generator,
             validation_data=validation_generator,
             epochs=2,
             callbacks=[earlystopping])

# ========================================
# Get the optimal hyperparameters
# ========================================
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(test_generator, validation_data=validation_generator, epochs=50)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# ========================================
# Retrain the model
# ========================================
hypermodel.fit(test_generator, validation_data=validation_generator, epochs=best_epoch)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# ========================================
# Visualising
# ========================================
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuaracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# ========================================
# Saving
# ========================================
tf.keras.models.save_model(hypermodel, 'tune.hdf5')
