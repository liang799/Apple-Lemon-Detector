from functions import image_gen_w_aug
import tensorflow as tf
import os

train_dir = os.path.join('C:/Python/rps/datasets/train/')
test_dir = os.path.join('C:/Python/rps/datasets/test/')

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir)

model = tf.keras.models.Sequential([
    # input is 75x75 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(75, 75, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
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
