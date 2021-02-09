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
dependents_train = np.loadtxt('train_dependents.csv', delimiter=',')
subjects_val = np.loadtxt('val_subjects.csv', delimiter=',')
scenes_val = np.loadtxt('val_scene_objects.csv', delimiter=',')
dependents_val = np.loadtxt('val_dependents.csv', delimiter=',')

print(subjects_train.shape)
print(scenes_train.shape)
print(dependents_train.shape)

max_scene_length = 14

training_scenes = []
training_outputs = []
validation_scenes = []
validation_outputs = []

for i in range(0, len(scenes_train)):
    object_list = fun.list_from_one_hot_encode(vector=scenes_train[i], possible_objects=scene_objects)
    # Add the option that the dependent is not in the scene
    object_list.append('none')
    training_scenes.append(object_list)

for i in range(0, len(scenes_val)):
    object_list = fun.list_from_one_hot_encode(vector=scenes_val[i], possible_objects=scene_objects)
    # Add the option that the dependent is not in the scene
    object_list.append('none')
    validation_scenes.append(object_list)

scene_vectors_train = fun.create_value_tensor(max_scene_length + 1, scenes=training_scenes,  word2vec=word2vec)
scene_vectors_val = fun.create_value_tensor(max_scene_length + 1, scenes=validation_scenes,  word2vec=word2vec)

# For each training scene
for i in range(0, len(scene_vectors_train)):
    dependent_index = max_scene_length
    dependent_vector = dependents_train[i]
    scene_matrix = scene_vectors_train[i]
    # See if the dependent matches any objects in the scene
    for j in range(0, len(scene_matrix)):
        word_vector = scene_matrix[j]
        if np.array_equal(dependent_vector, word_vector):
            dependent_index = j
    output_vector = np.zeros(shape=(1, max_scene_length + 1))
    output_vector[0, dependent_index] = 1
    training_outputs.append(output_vector)

# For each validation scene
for i in range(0, len(scene_vectors_val)):
    dependent_index = max_scene_length
    dependent_vector = dependents_val[i]
    scene_matrix = scene_vectors_val[i]
    # See if the dependent matches any objects in the scene
    for j in range(0, len(scene_matrix)):
        word_vector = scene_matrix[j]
        if np.array_equal(dependent_vector, word_vector):
            dependent_index = j
    output_vector = np.zeros(shape=(1, max_scene_length + 1))
    output_vector[0, dependent_index] = 1
    validation_outputs.append(output_vector)


x_train = [subjects_train, scene_vectors_train]
y_train = np.reshape(np.asarray(training_outputs), newshape=(len(training_outputs), max_scene_length + 1))
x_val = [subjects_val, scene_vectors_val]
y_val = np.reshape(np.asarray(validation_outputs), newshape=(len(validation_outputs), max_scene_length + 1))


'''
MODEL
'''
subject_input = tf.keras.Input(shape=(300,))
scene_input = tf.keras.Input(shape=(max_scene_length + 1, 300,))
inputs = [subject_input, scene_input]

batch_size = 32
subject_dense = layers.Dense(32, activation='relu')(inputs[0])
scene_dense = layers.Dense(32, activation='relu')(inputs[1])
subject_dense = tf.expand_dims(subject_dense, axis=1)
subject_dense_repeat = tf.repeat(subject_dense, repeats=[max_scene_length + 1], axis=1)
hidden1 = tf.concat([subject_dense_repeat, scene_dense], axis=2)
dropout1 = layers.Dropout(0.5)(hidden1)
hidden2 = layers.Dense(32, activation='relu')(dropout1)
dropout2 = layers.Dropout(0.5)(hidden2)
scores = layers.Dense(1, activation='relu')(dropout2)
outputs = tf.squeeze(scores, axis=2)
model = models.Model(inputs, outputs)

print(model.summary())

# Selecting the type of loss function and optimizer
model.compile(optimizer='SGD', loss=tf.nn.softmax_cross_entropy_with_logits)

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


num_epochs = 1000
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[model_checkpoint_callback])


# Loading the Best Weights
model.load_weights(checkpoint_filepath)

# Save this Model
model.save_weights('dependent_prediction_model')


# Training History
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']


num_epochs = len(val_loss_values)

# Plot the loss
fun.plot_loss(num_epochs, loss_values, val_loss_values)

'''
model.load_weights('dependent_prediction_model')


perplexity = fun.calculate_perplexity_with_attention(x_val, y_val, model)
print(perplexity)

subject = 'lightning'
object_list = ['ball', 'bat', 'sun', 'boy', 'none']

subject_vector = word2vec[subject]
subject_vector = tf.reshape(subject_vector, shape=(1, 300))
scene_array = fun.create_value_tensor(max_scene_length + 1, scenes=[object_list], word2vec=word2vec)
prediction = model.predict([subject_vector, scene_array])
softmax_layer = layers.Softmax()
prediction = softmax_layer(prediction)
prediction = prediction.numpy()[0]
prediction = list(prediction)
print(prediction)
max_index = prediction.index(max(prediction))
print(object_list[max_index])
