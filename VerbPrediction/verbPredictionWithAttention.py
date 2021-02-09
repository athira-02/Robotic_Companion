import numpy as np
import tensorflow as tf
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from tensorflow.keras import models
from tensorflow.keras import layers
import functions as fun
import attentionLayer as attention


scene_objects = ['helicopter', 'balloon', 'cloud', 'sun', 'lightning', 'rain', 'rocket', 'airplane', 'bouncy',
                 'slide', 'sandbox', 'grill', 'swing', 'tent', 'table', 'tree', 'tree', 'tree', 'boy', 'girl',
                 'bear', 'cat', 'dog', 'duck', 'owl', 'snake', 'hat', 'hat', 'hat', 'hat', 'hat', 'hat', 'hat',
                 'hat', 'glasses', 'glasses', 'pie', 'pizza', 'hotdog', 'ketchup', 'mustard', 'hamburger', 'soda',
                 'baseball', 'pail', 'ball', 'ball', 'ball', 'ball', 'ball', 'frisbee', 'bat', 'balloons', 'glove',
                 'shovel', 'racket', 'kite', 'fire']

# Word2Vec
word2vec = KeyedVectors.load_word2vec_format(
    datapath("//home/athira/Robotic_Companion/VerbPrediction/word2vec_vectors.bin"), binary=True)  # C bin format

'''
LOAD DATA
'''
subjects_train = np.loadtxt('train_subjects.csv', delimiter=',')
scenes_train = np.loadtxt('train_scene_objects.csv', delimiter=',')
verbs_train = np.loadtxt('train_verbs.csv', delimiter=',')
subjects_val = np.loadtxt('val_subjects.csv', delimiter=',')
scenes_val = np.loadtxt('val_scene_objects.csv', delimiter=',')
verbs_val = np.loadtxt('val_verbs.csv', delimiter=',')
'''
subjects_test = np.loadtxt('test_subjects.csv', delimiter=',')
scenes_test = np.loadtxt('test_scene_objects.csv', delimiter=',')
verbs_test = np.loadtxt('test_verbs.csv', delimiter=',')
'''

print(subjects_train.shape)
print(scenes_train.shape)
print(verbs_train.shape)

# Query Vectors
query_vectors_train = tf.convert_to_tensor(subjects_train)
query_vectors_train = tf.reshape(query_vectors_train, shape=(len(subjects_train), 300))
query_vectors_val = tf.convert_to_tensor(subjects_val)
query_vectors_val = tf.reshape(query_vectors_val, shape=(len(subjects_val), 300))

max_scene_length = 14

training_scenes = []
validation_scenes = []

for i in range(0, len(scenes_train)):
    object_list = fun.list_from_one_hot_encode(vector=scenes_train[i], possible_objects=scene_objects)
    training_scenes.append(object_list)
for i in range(0, len(scenes_val)):
    object_list = fun.list_from_one_hot_encode(vector=scenes_val[i], possible_objects=scene_objects)
    validation_scenes.append(object_list)

value_vectors_train = fun.create_value_tensor(max_scene_length, scenes=training_scenes,  word2vec=word2vec)
value_vectors_val = fun.create_value_tensor(max_scene_length, scenes=validation_scenes,  word2vec=word2vec)

'''
MODEL
'''
x_train = [query_vectors_train, value_vectors_train]
y_train = verbs_train
x_val = [query_vectors_val, value_vectors_val]
y_val = verbs_val


batch_size = 32

query_input = tf.keras.Input(shape=(300,))
value_input = tf.keras.Input(shape=(14, 300,))
inputs = [query_input, value_input]


attention_result, attention_weights = attention.BahdanauAttention(512)(query=inputs[0], values=inputs[1])
dropout1 = layers.Dropout(0.5, input_shape=(300,))(attention_result)
dense1 = layers.Dense(512, activation='relu')(dropout1)
dropout2 = layers.Dropout(0.5)(dense1)
dense2 = layers.Dense(512, activation='relu')(dropout2)
dropout3 = layers.Dropout(0.5)(dense2)
dense3 = layers.Dense(512, activation='relu')(dropout3)
dropout4 = layers.Dropout(0.5)(dense3)
outputs = layers.Dense(69)(dropout4)

model = models.Model(inputs, outputs)


# Selecting the type of loss function and optimizer
model.compile(optimizer='SGD', loss=tf.nn.softmax_cross_entropy_with_logits)

print(model.summary())

# Checkpoint to save weights at lowest validation loss
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


num_epochs = 2000
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[model_checkpoint_callback])


# Loading the Best Weights
model.load_weights(checkpoint_filepath)

# Save this Model
model.save_weights('verb_prediction_model_with_attention')


# Training History
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']


num_epochs = len(val_loss_values)

# Plot the loss
fun.plot_loss(num_epochs, loss_values, val_loss_values)


# Loads weights
model.load_weights('verb_prediction_model_with_attention')  # Load the model


# Calculate Perplexity
perplexity = fun.calculate_perplexity_with_attention(x_val, y_val, model)
print("Perplexity: " + str(perplexity))

