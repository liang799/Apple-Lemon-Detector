from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model


def image_gen_w_aug(train_parent_directory, valid_test_parent_directory, test_parent_directory):
    train_datagen = ImageDataGenerator(rescale=1 / 255,
                                       rotation_range=30,
                                       zoom_range=0.2,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)
    test_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                        target_size=(75, 75),
                                                        batch_size=214,
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
