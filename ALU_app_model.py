from functions import image_gen_w_aug, plot_confusion_matrix
from keras import callbacks, regularizers
import tensorflow as tf
import os
import matplotlib.pyplot as plt

train_dir = os.path.join('datasets/train/')
val_dir = os.path.join('datasets/val/')
test_dir = os.path.join('datasets/test/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, val_dir, test_dir)
# print(test_generator.classes.size)

model = tf.keras.models.Sequential([
    # input is 75x75 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(75, 75, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Normalize
    tf.keras.layers.BatchNormalization(),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)
history = model.fit(
    train_generator,
    epochs=25,
    batch_size=64,
    verbose=1,
    validation_data=validation_generator,
    callbacks=[earlystopping])

model.trainable = False

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

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

print("Do you wish to save the model?     (y/N):  ")
answer = input()
if answer == 'y' or answer == 'Y':
    tf.keras.models.save_model(model, 'model/my_model.hdf5')
