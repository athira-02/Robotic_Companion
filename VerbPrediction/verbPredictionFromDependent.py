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
dependents_train = np.loadtxt('train_dependents.csv', delimiter=',')
subjects_val = np.loadtxt('val_subjects.csv', delimiter=',')
dependents_val = np.loadtxt('val_dependents.csv', delimiter=',')

y_train = np.loadtxt('train_verbs.csv', delimiter=',')
y_val = np.loadtxt('val_verbs.csv', delimiter=',')


x_train = np.concatenate([subjects_train, dependents_train], axis=1)
x_val = np.concatenate([subjects_val, dependents_val], axis=1)

batch_size = 32

model = models.Sequential()
model.add(layers.Dropout(0.5, input_shape=(600,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(69))

# Selecting the type of loss function and optimizer
model.compile(optimizer='SGD', loss=tf.nn.softmax_cross_entropy_with_logits)

'''
TRAINING THE MODEL
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
model.save_weights('verb_prediction_from_dependent_model')


# Training History
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']


num_epochs = len(val_loss_values)

# Plot the loss
fun.plot_loss(num_epochs, loss_values, val_loss_values)

# Calculate Perplexity
model.load_weights('verb_prediction_from_dependent_model')  # Load the model

perplexity = fun.calculate_perplexity(x_val, y_val, model)
print("Perplexity: " + str(perplexity))
