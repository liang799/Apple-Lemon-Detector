from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
import itertools


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


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
