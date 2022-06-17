from sklearn.model_selection import GridSearchCV
from functions import image_gen_w_aug, plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from keras import callbacks, regularizers
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/train/')
val_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/val/')
test_dir = os.path.join('C:/Python/Apple-Lemon-Detector/datasets/test/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, val_dir, test_dir)

model = tf.keras.models.Sequential([
    # input is 75x75 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(75, 75, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.7),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.4),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    # Normalize
    # tf.keras.layers.BatchNormalization(),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


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


# Get classes
target_names = []
for key in train_generator.class_indices:
    target_names.append(key)

# Plot Confusion Matrix
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(test_generator.classes, y_pred)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

# Print Classification Report
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

# tf.keras.models.save_model(model, 'my_model.hdf5')