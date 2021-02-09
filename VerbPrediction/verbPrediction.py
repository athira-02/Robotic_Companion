import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import functions as fun


'''
LOAD DATA
'''
train_scene_sum = np.loadtxt('train_scene_sums.csv', delimiter=',')
train_subject = np.loadtxt('train_subjects.csv', delimiter=',')
val_scene_sum = np.loadtxt('val_scene_sums.csv', delimiter=',')
val_subject = np.loadtxt('val_subjects.csv', delimiter=',')

x_train = np.concatenate((train_scene_sum, train_subject), axis=1)
y_train = np.loadtxt('train_verbs.csv', delimiter=',')
x_val = np.concatenate((val_scene_sum, val_subject), axis=1)
y_val = np.loadtxt('val_verbs.csv', delimiter=',')


print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)


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


num_epochs = 2000
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[model_checkpoint_callback])


# Loading the Best Weights
model.load_weights(checkpoint_filepath)

# Save this Model
model.save_weights('verb_prediction_model')


# Training History
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']


num_epochs = len(val_loss_values)

# Plot the loss
fun.plot_loss(num_epochs, loss_values, val_loss_values)

# Calculate Perplexity
model.load_weights('verb_prediction_model')  # Load the model

perplexity = fun.calculate_perplexity(x_val, y_val, model)
print("Perplexity: " + str(perplexity))


