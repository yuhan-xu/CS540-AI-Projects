"""
Name: Yuhan Xu
Email: yxu329@wisc.edu
Class: CS 540
Project name: keras_intro.py
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


"""
takes an optional boolean argument and returns the data as described below
@:param training=True
@:return returns the data as described below
"""


def get_dataset(training=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    if training == True:  # if training is True, return the training images and labels as NumPy arrays
        np_training_images = np.array(train_images)
        np_training_labels = np.array(train_labels)
        return np_training_images, np_training_labels
    else:  # else,  return the testing images and labels
        return test_images, test_labels


# declare a global variable class_names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


"""
takes the dataset and labels produced by the previous function and prints several statistics about the data; does not 
return anything.
@:param images, labels
prints several statistics about the data
"""


def print_stats(images, labels):
    total_images = len(images)  # get length
    image_dim = str(len(images[0])) + "x" + str(len(images[0][0]))  # get dimension

    # print number of images and image dimension
    print(total_images)
    print(image_dim)

    number_list = [0] * len(class_names)  # create a list of zero and the length is the same as that of class_names

    # count the number of image that belongs to each name in the class_names
    for i in range(len(class_names)):
        for element in labels:
            if element == i:
                number_list[i] = number_list[i] + 1

    # use for loop to print the desired output
    for i in range(len(class_names)):
        print(str(i) + ". " + str(class_names[i]) + " - " + str(number_list[i]))


"""
takes a single image as an array of pixels and displays an image; does not return anything.
@:param image, label
displays an image
"""


def view_image(image, label):
    plt.imshow(image)  # use imshow() to show image
    plt.title(label)  # add title
    plt.colorbar()  # add colorbar


"""
 takes no arguments and returns an untrained neural network as specified below
@:param 
@:return returns an untrained neural network as specified below
"""


def build_model():
    model = keras.Sequential()  # specify the model

    # add layers to the model
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10))

    # compile the model, using the adam optimizer (Links to an external site.), the sparse categorical cross-entropy
    # loss function (Links to an external site.) (with the from_logits parameter set to True), and 'accuracy' as the
    # only metric
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


"""
takes the model produced by the previous function and the images and labels produced by the first function and trains 
the data for T epochs; does not return anything
@:param model, images, labels, T

"""


def train_model(model, images, labels, T):
    model.fit(images, labels, epochs=T)


"""
takes the trained model produced by the previous function and the test image/labels, and prints the evaluation 
statistics as described below (displaying the loss metric value if and only if the optional parameter has not been set 
to False)
@:param model, images, labels, show_loss=True
prints the evaluation statistics as described below (displaying the loss metric value if and only if the optional 
parameter has not been set to False)
"""


def evaluate_model(model, images, labels, show_loss=True):
    test_loss, test_accuracy = model.evaluate(images, labels)  # get test_loss, test_accuracy

    # if show_loss is False, only print Accuracy
    if show_loss == False:
        print('Accuracy: ' + str("{:.2%}".format(test_accuracy)))
    else:  # else, print Loss and Accuracy
        print('Loss: ' + str("%.2f" % test_loss))
        print('Accuracy: ' + str("{:.2%}".format(test_accuracy)))


"""
takes the trained model and test images, and prints the top 3 most likely labels for the image at the given index, 
along with their probabilities
@:param model, images, index
prints the top 3 most likely labels for the image at the given index, along with their probabilities
"""


def predict_label(model, images, index):
    model_pred = keras.Sequential()  # specify model_pred
    model_pred.add(model)  # add model to model_pred
    model_pred.add(keras.layers.Softmax())  # add softmax
    p = model_pred.predict(images)  # use predict() to predict the model
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
        print(str(class_names[maximum]) + ": " + str("{:.2%}".format(v_max)))
        list1.remove(v_max)



