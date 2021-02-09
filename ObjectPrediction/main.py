import tensorflow as tf
import random
import csv
import numpy as np
from scipy import spatial
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import math

from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.regularizers import l1

import functions as fun
from numpy import savetxt
from numpy import loadtxt


# Word2Vec
word2vec = KeyedVectors.load_word2vec_format(
    datapath("//home/athira/Robotic_Companion/ObjectPrediction/word2vec_vectors.bin"), binary=True)  # C bin format

# Edits made to sketchy scene words: street lamp--> lamp and picnic rug--> rug
sketchy_scene_words = ['words', 'airplane', 'apple', 'balloon', 'banana', 'basket', 'bee', 'bench', 'bicycle', 'bird',
                       'bottle',
                       'bucket', 'bus', 'butterfly', 'car', 'cat', 'chair', 'chicken', 'cloud', 'cow', 'cup',
                       'dinnerware', 'dog',
                       'duck', 'fence', 'flower', 'grape', 'grass', 'horse', 'house', 'moon', 'mountain', 'people',
                       'rug',
                       'pig', 'rabbit', 'road', 'sheep', 'sofa', 'star', 'lamp', 'sun', 'table', 'tree', 'truck',
                       'umbrella', 'others']

# List of word2vec vectors for each sketchy scene word... shifted by 1 so that airplane --> 0
word_vectors = []
for i in range(1, len(sketchy_scene_words) - 1):
    vector = word2vec[sketchy_scene_words[i]]
    word_vectors.append(vector)

number_of_training_samples = 5617
number_of_validation_samples = 535

training_data_path = '../ObjectPredictionData/training_data.csv'
validation_data_path = '../ObjectPredictionData/validation_data.csv'


'''
Uncomment the following lines to prepare data (create training, validation, and test data)
'''
# fun.data_preparation(training_data_path, number_of_training_samples, sketchy_scene_words, word2vec, input_fname='training_input.csv', labels_fname='training_labels.csv')
# fun.data_preparation(validation_data_path, number_of_validation_samples, sketchy_scene_words, word2vec, input_fname='validation_input.csv', labels_fname='validation_labels.csv')

# fun.expand_dataset_prep(training_data=training_data_path, input_fname='expanded_training_input.csv', labels_fname='expanded_training_labels.csv',
#                        sample_number_fname='sample_numbers.csv', sketchy_scene_words=sketchy_scene_words, number_of_samples=number_of_training_samples,
#                        word2vec=word2vec)


'''
CUSTOM LAYERS
'''


# Cosine Similarity Layer
class CosineSimilarity(tf.keras.layers.Layer):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def call(self, inputs):
        similarity_list = []
        predicted_vector = inputs.numpy()  # Convert Input Tensor to a numpy array
        for i in range(0, len(word_vectors)):
            cosine_similarity = 1 - spatial.distance.cosine(predicted_vector, word_vectors[i])
            similarity_list.append(cosine_similarity)
        return similarity_list


# This layer multiplies the matrix of word vectors (from sketchy_scene) with the input 300 dimension vector (matrix W)
class WordVectorLayer(layers.Layer):

    def __init__(self, **kwargs):
        super(WordVectorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], 45))

        w_np_matrix = np.zeros(shape=(len(word_vectors), 300), dtype="float32")

        for i in range(0, len(word_vectors)):
            w_np_matrix[i, :] = word_vectors[i]

        w_matrix = tf.convert_to_tensor(w_np_matrix)

        self.W = tf.reshape(w_matrix, shape=shape)

        super(WordVectorLayer, self).build(input_shape)

    def call(self, inputs):
        y = tf.matmul(inputs, self.W)
        return y


# This layer has a trainable projection matrix. It projects the 300 dimension word vectors to a lower dimension
# It then multiplies the projections by the input vector
class ProjectVectorLayer(layers.Layer):
    def __init__(self, dimension, **kwargs):
        self.dimension = dimension
        super(ProjectVectorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = tf.TensorShape((input_shape[1], 45))

        # Create a trainable projection matrix... Projects the 300 dimension word vectors to a lower dimension
        self.projection = self.add_weight(name='projection',
                                      shape=(self.dimension, 300),
                                      initializer='uniform',
                                      trainable=True)



        super(ProjectVectorLayer, self).build(input_shape)

    def call(self, inputs):
        # 300 Dimension Word Vector Matrix
        w_np_matrix = np.zeros(shape=(len(word_vectors), 300), dtype="float32")

        for i in range(0, len(word_vectors)):
            w_np_matrix[i, :] = word_vectors[i]

        w_matrix = tf.convert_to_tensor(w_np_matrix)

        # Project each word to lower dimension
        projected_matrix = tf.transpose(tf.matmul(self.projection, tf.transpose(w_matrix)))

        self.W = tf.reshape(projected_matrix, shape=self.shape)

        y = tf.matmul(inputs, self.W)
        return y


# Added Softmax layer to change model output into a probability distribution
softmax_layer = tf.keras.layers.Softmax()
cosine_similarity_layer = CosineSimilarity()

'''
DATA
'''
# x_train = loadtxt('training_input.csv', delimiter=',')
# y_train = loadtxt('training_labels.csv', delimiter=',')
x_train = loadtxt('expanded_training_input.csv', delimiter=',')
y_train = loadtxt('expanded_training_labels.csv', delimiter=',')
x_val = loadtxt('validation_input.csv', delimiter=',')
y_val = loadtxt('validation_labels.csv', delimiter=',')


batch_size = 32

# fun.create_vector_output_labels(y_train=y_train, y_val=y_val, training_fname='training_vector_labels.csv', validation_fname='validation_vector_labels.csv', word_vectors=word_vectors)

vector_y_train = loadtxt('training_vector_labels.csv', delimiter=',')
vector_y_val = loadtxt('validation_vector_labels.csv', delimiter=',')


'''
MODEL 1 - Input --> Output no intermediate layers
'''

single_layer_model = models.Sequential()
single_layer_model.add(layers.Dense(45, input_shape=(300,)))

# Selecting the type of loss function and optimizer
single_layer_model.compile(optimizer='SGD', loss=tf.nn.softmax_cross_entropy_with_logits)


'''
MODEL 2 Input --> Dense Layer --> Output (Drop Off Layers between)
'''
multi_layer_model = models.Sequential()
multi_layer_model.add(Dropout(0.5, input_shape=(300,)))
multi_layer_model.add(layers.Dense(128, activation='relu'))
multi_layer_model.add(Dropout(0.5))
multi_layer_model.add(layers.Dense(45))


# Selecting the type of loss function and optimizer
multi_layer_model.compile(optimizer='SGD', loss=tf.nn.softmax_cross_entropy_with_logits)

'''
MODEL 3 Input --> Word Vector (with Cosine Similarity Loss)
'''
cos_sim_model = models.Sequential()
cos_sim_model.add(layers.Dense(128, activation='relu', input_shape=(300,)))
cos_sim_model.add(layers.Dense(300))  # Output Layer is 300 dimension vector

# Selecting the type of loss function and optimizer
cos_sim_model.compile(optimizer='SGD', loss=tf.keras.losses.CosineSimilarity())  # Training With Cosine Similarity Loss Function

'''
MODEL 4 Input --> Dense Layer --> Word Vector --> Multiply by W --> Output
'''
word_vector_model = models.Sequential()
word_vector_model.add(layers.Dense(64, activation='relu', input_shape=(300,)))
word_vector_model.add(layers.Dense(300))
word_vector_model.add(layer=WordVectorLayer())


word_vector_model.compile(optimizer='SGD', loss=tf.nn.softmax_cross_entropy_with_logits, run_eagerly=True)


'''
MODEL 5 Input --> Dense Layer --> Projection --> Output
'''
model = models.Sequential()
model.add(layers.Dense(32, input_shape=(300,)))
model.add(layer=ProjectVectorLayer(dimension=32))

model.compile(optimizer='SGD', loss=tf.nn.softmax_cross_entropy_with_logits, run_eagerly=True)


'''
TRAINING THE MODEL
'''
'''
# Checkpoint to save weights at lowest validation loss
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300)


num_epochs = 250
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(x_val, y_val), callbacks=[model_checkpoint_callback])

# Training History
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

num_epochs = len(val_loss_values)

# Plot the loss
fun.plot_loss(num_epochs, loss_values, val_loss_values)

# Loading the Best Weights
model.load_weights(checkpoint_filepath)

# Save this Model
model.save_weights('projection_model')
'''

'''
LOADING MODELS
'''

single_layer_model.load_weights('single_layer_model')
multi_layer_model.load_weights('multi_layer_model')
cos_sim_model.load_weights('cos_sim_model')
word_vector_model.load_weights('word_vector_model')
model.load_weights('projection_model')


'''
PERPLEXITY
'''
'''
single_layer_perplexity = fun.calculate_perplexity(x_val=x_val, y_val=y_val, model=single_layer_model, extra_layers=[softmax_layer])
print("Single Layer Perplexity: " + str(single_layer_perplexity))
multi_layer_perplexity = fun.calculate_perplexity(x_val=x_val, y_val=y_val, model=multi_layer_model, extra_layers=[softmax_layer])
print("Multi Layer Perplexity: " + str(multi_layer_perplexity))
cos_sim_perplexity = fun.calculate_perplexity(x_val=x_val, y_val=y_val, model=cos_sim_model, extra_layers=[cosine_similarity_layer, softmax_layer])
print("Cos Sim Perplexity: " + str(cos_sim_perplexity))
word_vector_perplexity = fun.calculate_perplexity(x_val=x_val, y_val=y_val, model=word_vector_model, extra_layers=[softmax_layer])
print("Word Vec Perplexity: " + str(word_vector_perplexity))
projection_perplexity = fun.calculate_perplexity(x_val=x_val, y_val=y_val, model=model, extra_layers=[softmax_layer])
print("Perplexity: " + str(projection_perplexity))
'''


# Getting list of objects from SpeechBlocks (needs "reference-dictionary.txt" (list of speechblocks sprites)
with open('reference-dictionary.txt') as fp:
    lines = fp.readlines()


possible_objects = []

for line in lines:
    line_split = line.split()
    split_1 = line_split[0]

    if len(line_split) > 1:
        split_2 = line_split[1]
        second_split = split_2.split(sep=',')

        for word in second_split:
            the_object = ''
            word_complete = False
            for character in word:
                if character == '_' or character == '.':
                    word_complete = True
                if not word_complete:
                    the_object = the_object + character
            if the_object not in possible_objects and the_object in word2vec.vocab:
                possible_objects.append(the_object)

print(len(possible_objects))

possible_vectors = []

for possible_object in possible_objects:
    vector = word2vec[possible_object]
    possible_vectors.append(vector)


'''
PREDICTIONS (testing for specific scenes)
'''
'''
# Test for selected scene
scene = ['throne', 'king', 'crown', 'queen'] #Scene
print(scene)
input_vector = fun.create_scene_vector(scene, word2vec)

vector_output_model = models.Sequential()
vector_output_model.add(layers.Dense(64, activation='relu', input_shape=(300,)))
vector_output_model.add(layers.Dense(300))

vector_output_model.load_weights('word_vector_model')


# Using vector outputs to predict a verb
output_vector = vector_output_model.predict(input_vector)[0]
print('Word2Vec prediction: ' + str(word2vec.most_similar(positive=[output_vector, ])[0][0]))

cos_sim_output_vector = cos_sim_model.predict(input_vector)[0]
print('Cos sim Word2Vec prediction: ' + str(word2vec.most_similar(positive=[cos_sim_output_vector, ])[0][0]))


print('Projection Model Prediction: ' + str(fun.predict_object(model=model, scene=scene, sketchy_scene_words=sketchy_scene_words, word2vec=word2vec)))

print('Vector output prediction: ' + str(fun.predict_object(model=word_vector_model, scene=scene, sketchy_scene_words=sketchy_scene_words, word2vec=word2vec)))

similarities_to_output_vector = word2vec.cosine_similarities(output_vector, np.array(possible_vectors))

similarities_to_output_vector = np.ndarray.tolist(similarities_to_output_vector)


sim_to_cos_output = word2vec.cosine_similarities(cos_sim_output_vector, np.array(possible_vectors))
sim_to_cos_output = np.ndarray.tolist(sim_to_cos_output)

sim_to_cos_output_sketchy_scene = word2vec.cosine_similarities(cos_sim_output_vector, np.array(word_vectors))
sim_to_cos_output_sketchy_scene = np.ndarray.tolist(sim_to_cos_output_sketchy_scene)


max_sim = 0
max_sim_index = 0
for j in range(0, len(similarities_to_output_vector)):
    if similarities_to_output_vector[j] > max_sim:
        max_sim = similarities_to_output_vector[j]
        max_sim_index = j
print('SpeechBlocks word vector prediction: ' + str(possible_objects[max_sim_index]))

max_sim = 0
max_sim_index = 0
for j in range(0, len(sim_to_cos_output)):
    if sim_to_cos_output[j] > max_sim:
        max_sim = sim_to_cos_output[j]
        max_sim_index = j
print('Cos sim SpeechBlocks word vector prediction: ' + str(possible_objects[max_sim_index]))

max_sim = 0
max_sim_index = 0
for j in range(0, len(sim_to_cos_output_sketchy_scene)):
    if sim_to_cos_output_sketchy_scene[j] > max_sim:
        max_sim = sim_to_cos_output_sketchy_scene[j]
        max_sim_index = j
print('Cos sim prediction: ' + str(sketchy_scene_words[max_sim_index + 1]))
'''
