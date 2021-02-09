import nltk
import lemmatization
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import math
import csv
import numpy as np


# One hot encode a list
def one_hot_encode(object_list, possible_objects):
    output_vector = np.zeros(shape=(len(possible_objects)))
    for i in range(0, len(object_list)):
        if object_list[i] in possible_objects:
            object_index = possible_objects.index(object_list[i])
            output_vector[object_index] = 1
    return output_vector


# Get a list from a one hot encoded vector
def list_from_one_hot_encode(vector, possible_objects):
    object_list = []
    for j in range(0, len(vector)):
        if int(vector[j]) == 1:
            word = possible_objects[j]
            if word not in object_list:
                object_list.append(word)
    return object_list


# Parse and Lemmatize a Relation to get a verb
def relation_to_verb(relation):
    # Remove unnecessary characters
    new_string = ''
    for i in range(0, len(relation)):
        if relation[i] != '(' and relation[i] != ')':
            new_string = new_string + relation[i]
    relation = new_string

    # Parse out the verb
    relation_tag = (nltk.pos_tag(relation.split()))
    verb = ''
    for j in range(0, len(relation_tag)):
        word = relation_tag[j][0]
        part_of_speech = relation_tag[j][1]
        if "VB" in part_of_speech and word != 'sandbox':
            verb = word
        elif "NNS" in part_of_speech and word != 'towards':
            verb = word
    if verb == '':
        return verb
    else:
        # Lemmatize the verb
        lemmatizer = lemmatization.AdjustedLemmatizer()
        lemmatized_word = lemmatizer.lemmatize(verb, wordnet.VERB)

        return lemmatized_word


def create_scene_object_list(file, scene_objects):
    scene_object_instances = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            object_list = row
            del object_list[-1]
            scene_object_instances.append(object_list)

    scene_object_list = []
    for i in range(0, len(scene_object_instances)):
        scene = scene_object_instances[i]
        object_list = list_from_one_hot_encode(vector=scene, possible_objects=scene_objects)
        scene_object_list.append(object_list)

    return scene_object_list


# Create the tensor for the value for the attention layer. This is a tensor of shape (batch_size, max_scene_length, 300)
def create_value_tensor(max_scene_length, scenes,  word2vec):
    value_vectors = tf.zeros(shape=(0, max_scene_length, 300))
    for i in range(0, len(scenes)):
        scene = scenes[i]
        scene_matrix = np.zeros(shape=(max_scene_length, 300))
        for j in range(0, len(scene)):
            vector = word2vec[scene[j]]
            scene_matrix[j] = vector
        scene_tensor = tf.convert_to_tensor(scene_matrix)
        scene_tensor = tf.reshape(scene_tensor, shape=(1, max_scene_length, 300))
        scene_tensor = tf.cast(scene_tensor, dtype=float)
        value_vectors = tf.concat(values=(value_vectors, scene_tensor), axis=0)

    return value_vectors


# Calculates the Perplexity for the model
def calculate_perplexity(x_val, y_val, model):
    perplexity_sum = 0
    for i in range(0, len(y_val)):
        input_vector = x_val[i: i + 1]
        prob_distr = model.predict(input_vector)[0]  # Prediction for this sample

        soft_max_layer = layers.Softmax()
        prob_distr = soft_max_layer(prob_distr)

        label = y_val[i]
        verb_index = np.where(label == 1)[0][0]

        p = prob_distr[verb_index]  # Probability of the correct object
        perplexity_sum += math.log(p, 2)

    exponent = (-perplexity_sum / len(y_val))
    perplexity = 2 ** exponent

    return perplexity


# Calculates the Perplexity for the attention model (x val is [query, value])
def calculate_perplexity_with_attention(x_val, y_val, model):
    perplexity_sum = 0
    val_query = x_val[0]
    val_value = x_val[1]
    for i in range(0, len(y_val)):
        query_vector = val_query[i: i + 1]
        value_vector = val_value[i: i + 1]
        prob_distr = model.predict([query_vector, value_vector])[0]  # Prediction for this sample
        soft_max_layer = layers.Softmax()
        prob_distr = soft_max_layer(prob_distr)

        label = y_val[i]
        verb_index = np.where(label == 1)[0][0]

        p = prob_distr[verb_index]  # Probability of the correct object
        perplexity_sum += math.log(p, 2)

    exponent = (-perplexity_sum / len(y_val))
    perplexity = 2 ** exponent

    return perplexity


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

