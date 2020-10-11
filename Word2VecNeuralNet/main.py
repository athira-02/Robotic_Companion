import tensorflow as tf
import random
import csv
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import math
import functions as fun
from numpy import savetxt
from numpy import loadtxt

word2vec = KeyedVectors.load_word2vec_format(
    datapath("//home/athira/PycharmProjects/BasicWord2VecNeuralNet/word2vec_vectors.bin"), binary=True)  # C bin format

# Edits made to sketchy scene words: street lamp--> lamp and picnic rug--> rug
sketchy_scene_words = ['words', 'airplane', 'apple', 'balloon', 'banana', 'basket', 'bee', 'bench', 'bicycle', 'bird',
                       'bottle',
                       'bucket', 'bus', 'butterfly', 'car', 'cat', 'chair', 'chicken', 'cloud', 'cow', 'cup',
                       'dinnerware', 'dog',
                       'duck', 'fence', 'flower', 'grape', 'grass', 'horse', 'house', 'moon', 'mountain', 'people',
                       'rug',
                       'pig', 'rabbit', 'road', 'sheep', 'sofa', 'star', 'lamp', 'sun', 'table', 'tree', 'truck',
                       'umbrella', 'others']

number_of_training_samples = 5617
number_of_validation_samples = 535

# fun.data_preparation('../Data/training_data.csv',number_of_training_samples, sketchy_scene_words, word2vec, input_fname='training_input.csv', labels_fname='training_labels.csv')
# fun.data_preparation('../Data/validation_data.csv', number_of_validation_samples, sketchy_scene_words, word2vec, input_fname='validation_input.csv', labels_fname='validation_labels.csv')


x_train = loadtxt('training_input.csv', delimiter=',')
y_train = loadtxt('training_labels.csv', delimiter=',')
x_val = loadtxt('validation_input.csv', delimiter=',')
y_val = loadtxt('validation_labels.csv', delimiter=',')

batch_size = 32

model = models.Sequential()
# model.add(layers.Dense(45, activation='softmax', input_shape=(300,)))
model.add(layers.Dense(45, input_shape=(300,)))

# Selecting the type of loss function and optimizer
model.compile(optimizer='SGD', loss=tf.nn.softmax_cross_entropy_with_logits)

early_stopping = EarlyStopping(verbose=1, patience=100)


# Training the data with 20 epochs (iterations) with mini-batches of 512 samples
num_epochs = 500
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])

# Training History
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

num_epochs = len(val_loss_values)

# Plot the loss
fun.plot_loss(num_epochs, loss_values, val_loss_values)


layer = tf.keras.layers.Softmax()

# Calculating the Perplexity
perplexity_sum = 0
for i in range(0, len(y_val)):
    input_vector = x_val[i: i+1]
    prob_distr = model.predict(input_vector)[0]  # Prediction for this sample
    prob_distr = layer(prob_distr).numpy()  # Softmax
    label = y_val[i]
    index = np.where(label == 1)[0][0]
    p = prob_distr[index]  # Probability of the correct object
    perplexity_sum += math.log(p, 2)


exponent = (-perplexity_sum/len(y_val))
perplexity = 2 ** exponent

print("Perplexity: " + str(perplexity))


