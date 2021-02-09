import random
import csv
import numpy as np
import tensorflow as tf
import math
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from numpy import savetxt
import matplotlib.pyplot as plt


# Expanding the Dataset to create n scenes for a scene with n objects
def expand_dataset_prep(training_data, input_fname, labels_fname, sample_number_fname, sketchy_scene_words, number_of_samples, word2vec):
    labels = []  # The removed object for each scene
    scenes = []  # The vector sum of the objects in a scene
    sample_numbers = []  # The sample number associated with each row of labels and scenes

    for sample_number in range(1, number_of_samples + 1):
        word_list = [] # The list of words in the scene

        with open(training_data) as fd:
            reader = csv.reader(fd)
            scene_list = [row for idx, row in enumerate(reader) if idx == sample_number - 1]

        scene_list = scene_list[0]
        del scene_list[0]

        for i in scene_list:
            if int(i) != 46:
                word_list.append(sketchy_scene_words[int(i)])

        for word in word_list:
            sample_numbers.append(sample_number)  # Associate this row of labels with the correct sample

            word_number = sketchy_scene_words.index(word) - 1  # Keys are shifted by 1 (1 = airplane is 0)
            label = np.zeros(shape=(1, 45))  # Create a 45 dimension vector
            label[0, word_number] = 1  # Put a 1 in the position of the correct word

            labels.append(label)  # Add the label to the output matrix

            scene = [x for x in word_list if x != word]  # The remaining words in the scene

            # Sum the remaining words in the scene
            word_sum = np.zeros(shape=(1, 300))
            for element in scene:
                word_sum = word_sum + word2vec[element]
            scenes.append(word_sum)  # Add the sum of the words to the input matrix

    # Saving the data
    training_data = np.zeros(shape=(len(scenes), 300))
    training_labels = np.zeros(shape=(len(labels), 45))
    training_sample_numbers = np.zeros(shape=(len(sample_numbers), 1))

    for i in range(0, len(scenes)):
        training_data[i, :] = scenes[i]
    for i in range(0, len(labels)):
        training_labels[i, :] = labels[i]
    for i in range(0, len(sample_numbers)):
        training_sample_numbers[i, :] = sample_numbers[i]

    savetxt(input_fname, training_data, delimiter=',')
    savetxt(labels_fname, training_labels, delimiter=',', fmt='%i')
    savetxt(sample_number_fname, training_sample_numbers, delimiter=',', fmt='%i')


def data_preparation(data,number_of_samples, sketchy_scene_words, word2vec, input_fname, labels_fname):
    training_scenes = []
    labels = []

    for sample_number in range(1, number_of_samples + 1):
        scene_list = []
        word_list = []

        with open(data) as fd:
            reader = csv.reader(fd)
            scene_list = [row for idx, row in enumerate(reader) if idx == sample_number - 1]

        scene_list = scene_list[0]
        del scene_list[0]

        for i in scene_list:
            if int(i) != 46:
                word_list.append(sketchy_scene_words[int(i)])

        # Remove an object randomly
        random_index = random.randint(0, len(word_list) - 1)

        removed_word = word_list.pop(random_index)
        removed_word_num = sketchy_scene_words.index(removed_word) - 1  # Keys are shifted by 1 (1 = airplane is 0)

        training_scenes.append(word_list)
        labels.append(removed_word_num)

    training_data = np.zeros(shape=(number_of_samples, 300))
    training_labels = np.zeros(shape=(number_of_samples, 45))
    for i in range(0, len(training_scenes)):
        word_sum = np.zeros(shape=(1, 300))
        for word in training_scenes[i]:
            word_sum = word_sum + word2vec[word]
        training_data[i, :] = word_sum[0, :]
        training_labels[i, labels[i]] = 1   # Put a 1 at the index of the correct word

    savetxt(input_fname, training_data, delimiter=',')
    savetxt(labels_fname, training_labels, delimiter=',', fmt='%i')


# Convert the training and validation labels to the correct word2vec vector
def create_vector_output_labels(y_train, y_val, training_fname, validation_fname, word_vectors):
    training_labels = []
    validation_labels = []
    for row in range(0, len(y_train)):
        index = get_removed_object_index(y_train, row)
        vec = word_vectors[index]
        training_labels.append(vec)
    for row in range(0, len(y_val)):
        index = get_removed_object_index(y_val, row)
        vec = word_vectors[index]
        validation_labels.append(vec)

    # Saving the Files
    training_data = np.zeros(shape=(len(training_labels), 300))
    validation_data = np.zeros(shape=(len(validation_labels), 300))

    for i in range(0, len(training_labels)):
        training_data[i, :] = training_labels[i]
    for i in range(0, len(validation_labels)):
        validation_data[i, :] = validation_labels[i]

    savetxt(training_fname, training_data, delimiter=',')
    savetxt(validation_fname, validation_data, delimiter=',')


# Get the list of object in the scene
def get_object_list(data, labels, sample_number, sketchy_scene_words):
    word_list = []
    with open(data) as fd:
        reader = csv.reader(fd)
        scene_list = [row for idx, row in enumerate(reader) if idx == sample_number - 1]
    scene_list = scene_list[0]
    del scene_list[0]

    for i in scene_list:
        if int(i) != 46:
            word_list.append(sketchy_scene_words[int(i)])

    # Get the word to remove
    removed_word = get_removed_object(labels, sample_number - 1, sketchy_scene_words)

    word_list.remove(removed_word)

    return word_list


# Get the removed object's index... this is shifted by 1 (i.e. airplane --> 0)
def get_removed_object_index(labels, row):
    label = labels[row]
    index = np.where(label == 1)[0][0]

    return index


# Get the removed object
def get_removed_object(labels, row, sketchy_scene_words):
    label = labels[row]
    index = np.where(label == 1)[0][0]

    word = sketchy_scene_words[index + 1]

    return word


# Plots the loss and accuracy of the training
def plot_loss_accuracy(epochs, loss_values, val_loss_values, acc_values, val_acc_values):
    epochs = range(1, epochs + 1)
    plt.plot(epochs, loss_values, 'ro', label='Training Loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    plt.plot(epochs, acc_values, 'go', label='Training Accuracy')
    plt.plot(epochs, val_acc_values, 'g', label="Validation Accuracy")
    plt.title('Training and Validation Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()


# Plots the loss of the training
def plot_loss(epochs, loss_values, val_loss_values):
    epochs = range(1, epochs + 1)
    plt.plot(epochs, loss_values, 'r', label='Training Loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Calculates the Perplexity for the model
def calculate_perplexity(x_val, y_val, model, extra_layers):
    perplexity_sum = 0
    for i in range(0, len(y_val)):
        input_vector = x_val[i: i + 1]
        prob_distr = model.predict(input_vector)[0]  # Prediction for this sample
        for layer in extra_layers:
            prob_distr = layer(prob_distr)
        index = get_removed_object_index(labels=y_val, row=i)  # Removed Word
        p = prob_distr[index]  # Probability of the correct object
        perplexity_sum += math.log(p, 2)

    exponent = (-perplexity_sum / len(y_val))
    perplexity = 2 ** exponent

    return perplexity


# Predict for a given scene
def predict_object(model, scene, sketchy_scene_words, word2vec):
    input_vector = create_scene_vector(scene, word2vec)
    output_word = model.predict(input_vector)[0]
    softmax_layer = tf.keras.layers.Softmax()
    output_word = softmax_layer(output_word)
    output_word = np.ndarray.tolist(output_word.numpy())
    max_index = 0
    max_prob = 0
    for i in range(0, len(output_word)):
        if output_word[i] > max_prob:
            max_prob = output_word[i]
            max_index = i

    return sketchy_scene_words[max_index + 1]

# Given a list of object is a scene, returns the sum of the vectors for these words
def create_scene_vector(scene, word2vec):
    word_sum = np.zeros(shape=(1, 300))
    for element in scene:
        word_sum = word_sum + word2vec[element]
    return word_sum
