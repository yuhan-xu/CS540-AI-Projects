"""
Name: Yuhan Xu
Email: yxu329@wisc.edu
Class: CS 540
Project name: keras_cnn.py
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# declare a global variable class_names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


"""
takes an optional boolean argument and returns the data as described below
@:param training=True
@:return returns the data as described below
"""


def get_dataset(training=True):
    # use the usual command to get the 3D array as did in P9
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # use NumPy's expand_dims() functions to add a single value to the images in the fourth dimension, to signify that
    # the images are in greyscale. You can just leave these values as their default (0)
    reshaped_train = np.expand_dims(train_images, axis=3)
    reshaped_test = np.expand_dims(test_images, axis=3)

    if training == True:  # if training is True, return the training images and labels as NumPy arrays
        np_training_images = np.array(train_images)
        np_training_labels = np.array(train_labels)
        return reshaped_train, np_training_labels
    else:  # else,  return the testing images and labels
        return reshaped_test, test_labels


"""
takes no arguments and returns an untrained neural network as specified below
@:param No parameters
@:return returns an untrained neural network as specified below
"""


def build_model():
    model = keras.Sequential()  # specify the model

    # add two 2D convolutional layers and Flatten and Dense layers to the modeol
    model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    # compile the model, using the adam optimizer, the categorical cross-entropy loss function, and 'accuracy' as the
    # only metric
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model  # return the model


"""
takes the model produced by the previous function and the images and labels produced by the first function and trains 
the data for T epochs; We're combining the training and evaluation steps this time; does not return anything
@:param model, train_img, train_lab, test_img, test_lab, T

"""


def train_model(model, train_img, train_lab, test_img, test_lab, T):

    # transform both train_lab and test_lab to a 2d array before passing to model.fit()
    train_lab = keras.utils.to_categorical(train_lab)
    test_lab = keras.utils.to_categorical(test_lab)

    # use model.fit()'s optional validation_data argument to fit the model
    model.fit(train_img, train_lab, validation_data=(test_img, test_lab), epochs=T)


"""
takes the trained model and test images, and prints the top 3 most likely labels for the image at the given index, 
along with their probabilities
@:param model, images, index
prints the top 3 most likely labels for the image at the given index, along with their probabilities
"""


def predict_label(model, images, index):
    p = model.predict(images)  # use predict() to predict the model
    predicted = p[index]
    list1 = []  # create an empty list named list1

    # append each predict[i] to a new list called list1
    for i in range(len(predicted)):
        list1.append(predicted[i])

    # use for loop to prints the top 3 most likely labels for the image at the given index, along with their
    # probabilities
    for j in range(3):
        v_max = 0
        for k in range(len(list1)):
            if list1[k] > v_max:
                v_max = list1[k]
        maximum = list1.index(v_max)
        print(str(class_names[maximum]) + ": " + str("{:.2%}".format(v_max)))  # adjust the format when printing
        list1.remove(v_max)
