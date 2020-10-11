import random
import csv
import numpy as np
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from numpy import savetxt
import matplotlib.pyplot as plt



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

