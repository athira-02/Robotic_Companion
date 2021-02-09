import numpy as np
import json
import random
import functions as fun
from gensim.test.utils import datapath
from gensim.models import KeyedVectors

abstract_scene_objects = ['helicopter', 'hotairballoon', 'cloud', 'sun', 'lightning', 'rain', 'rocket', 'airplane',
                          'bouncy', 'slide', 'sandbox', 'grill', 'swing', 'tent', 'table', 'pinetree', 'oaktree',
                          'appletree', 'boy', 'girl', 'bear', 'cat', 'dog', 'duck', 'owl', 'snake', 'baseballcap',
                          'crown', 'chefhat', 'piratehat', 'wintercap', 'bennie', 'wizardhat', 'vikinghat',
                          'purpleglasses', 'sunglasses', 'pie', 'pizza', 'hotdog', 'ketchup', 'mustard', 'hamburger',
                          'soda', 'baseball', 'pail', 'beachball', 'basketball', 'soccerball', 'tennisball',
                          'football', 'frisbee', 'baseballbat', 'balloons', 'baseballglove', 'shovel', 'tennisracket',
                          'kite', 'fire']

scene_objects = ['helicopter', 'balloon', 'cloud', 'sun', 'lightning', 'rain', 'rocket', 'airplane', 'bouncy',
                 'slide', 'sandbox', 'grill', 'swing', 'tent', 'table', 'tree', 'tree', 'tree', 'boy', 'girl',
                 'bear', 'cat', 'dog', 'duck', 'owl', 'snake', 'hat', 'hat', 'hat', 'hat', 'hat', 'hat', 'hat',
                 'hat', 'glasses', 'glasses', 'pie', 'pizza', 'hotdog', 'ketchup', 'mustard', 'hamburger', 'soda',
                 'baseball', 'pail', 'ball', 'ball', 'ball', 'ball', 'ball', 'frisbee', 'bat', 'balloons', 'glove',
                 'shovel', 'racket', 'kite', 'fire']

verb_list = ['play', 'sit','run', 'ask', 'wait', 'slide', 'burn', 'talk', 'climb', 'scare', 'jump', 'throw', 'enjoy',
             'fly', 'hold', 'make', 'laugh', 'set', 'swing', 'try', 'surprise', 'greet', 'camp', 'offer', 'fall',
             'warn', 'kick', 'remember', 'lick', 'wave', 'eat', 'hit', 'show', 'lose', 'steal', 'chase', 'cook',
             'wear', 'feed', 'stay', 'practice', 'retrieve', 'ruin', 'terrify', 'catch', 'find', 'protect', 'excite',
             'love', 'think', 'drop', 'upset', 'toss', 'park', 'attack', 'frighten', 'smile', 'hand', 'walk',
             'startle', 'call', 'shock', 'feast', 'threaten', 'pour', 'annoy', 'bat', 'blame', 'chat']

# Word2Vec
word2vec = KeyedVectors.load_word2vec_format(
    datapath("//home/athira/Robotic_Companion/AbstractScenePrediction/word2vec_vectors.bin"), binary=True)  # C bin format


# Open file with relationship information
with open('../VerbPredictionData/AbstractSceneRelationships.json') as f:
    data = json.load(f)

scene_list = data['scenes']

scene_data = []

for i in range(0, len(scene_list)):
    relationships = dict(scene_list[i])['relationships']
    for j in range(0, len(relationships)):
        relationship = dict(relationships[j])
        relation = relationship['relation'][0]
        verb = fun.relation_to_verb(relation)
        subject = relationship['subject']
        dependent = relationship['dependent']

        # Add this sentence if the verb is in the common verb list
        if verb != '' and verb in verb_list:
            scene_info = [str(i), subject, verb, dependent]
            scene_data.append(scene_info)


# Get list of objects in each scene
scene_object_list = fun.create_scene_object_list(file='../VerbPredictionData/AbstractSceneObjectOccurence.csv', scene_objects=scene_objects)

scene_word_list = []
scene_sum_list = []
subject_vector_list = []
dependent_list = []
verb_vector_list = []

# For each sentence
for i in range(0, len(scene_data)):
    scene_information = scene_data[i]
    sentence_number = int(scene_information[0])
    subject = scene_information[1]
    verb = scene_information[2]
    dependent = scene_information[3]
    subject_word = ''
    if subject in abstract_scene_objects:
        index = abstract_scene_objects.index(subject)
        subject_word = scene_objects[index]

    subject_vector = word2vec[subject]  # Subject Vector

    # Creating verb output vector (one hot encoded)
    verb_index = verb_list.index(verb)
    verb_vector = fun.one_hot_encode(object_list=[verb], possible_objects=verb_list)

    # Scenes corresponding to each sentence
    for scene_number in range((sentence_number * 10), (sentence_number * 10) + 10):
        object_list = scene_object_list[scene_number]
        word_list = []
        for word in object_list:
            if word != subject_word:
                word_list.append(word)

        scene_word_vector = fun.one_hot_encode(object_list=word_list, possible_objects=scene_objects)

        # Is the dependent in the scene? If not the dependent is "None"
        if dependent not in word_list:
            dependent = 'none'

        dependent_vector = word2vec[dependent]  # Dependent Vector

        # Sum objects in the scene
        word_sum = np.zeros(shape=300)
        for word in word_list:
            word_sum = word_sum + word2vec[word]

        # Add vectors to their lists only if they have not already been added and the scene is not of 0 length
        isRepeat = False
        for x in range(0, len(subject_vector_list)):
            if np.array_equal(subject_vector_list[x], subject_vector) and np.array_equal(scene_word_list[x], scene_word_vector):
                if np.array_equal(verb_vector_list[x], verb_vector):
                    isRepeat = True

        if not isRepeat and len(word_list) > 0:
            subject_vector_list.append(subject_vector)
            scene_word_list.append(scene_word_vector)
            scene_sum_list.append(word_sum)
            verb_vector_list.append(verb_vector)
            dependent_list.append(dependent_vector)
            print('Adding: ' + 'Subject: ' + subject + '  Scene: ' + str(word_list) + '  Verb: ' + verb + ' Dependent: ' + dependent)

# Convert to numpy arrays
scene_word_array = np.asarray(scene_word_list)
scene_sum_array = np.asarray(scene_sum_list)
verb_array = np.asarray(verb_vector_list)
dependent_array = np.asarray(dependent_list)
subject_array = np.asarray(subject_vector_list)

'''
Shuffling and sorting into validation,test, and training data
'''
# Shuffling to randomly assigning to test, validation, and training
shuffle_array = []
for i in range(0, len(subject_array)):
    shuffle_array.append(i)
random.shuffle(shuffle_array)

# Add Validation Data
val_subject = []
val_scene_words = []
val_scene_sum = []
val_dependent = []
val_verb = []

for i in range(0, 1000):
    x = shuffle_array[i]
    val_subject.append(subject_array[x])
    val_scene_words.append((scene_word_array[x]))
    val_scene_sum.append(scene_sum_array[x])
    val_dependent.append(dependent_array[x])
    val_verb.append(verb_array[x])

val_subject_array = np.asarray(val_subject)
val_scene_word_array = np.asarray(val_scene_words)
val_scene_sum_array = np.asarray(val_scene_sum)
val_dependent_array = np.asarray(val_dependent)
val_verb_array = np.asarray(val_verb)

np.savetxt('val_subjects.csv', val_subject_array, delimiter=',')
np.savetxt('val_scene_objects.csv', val_scene_word_array, delimiter=',')
np.savetxt('val_scene_sums.csv', val_scene_sum_array, delimiter=',')
np.savetxt('val_dependents.csv', val_dependent_array, delimiter=',')
np.savetxt('val_verbs.csv', val_verb_array, delimiter=',')

# Add Test Data
test_subject = []
test_scene_words = []
test_scene_sum = []
test_dependent = []
test_verb = []

for i in range(1000, 2000):
    x = shuffle_array[i]
    test_subject.append(subject_array[x])
    test_scene_words.append((scene_word_array[x]))
    test_scene_sum.append(scene_sum_array[x])
    test_dependent.append(dependent_array[x])
    test_verb.append(verb_array[x])

test_subject_array = np.asarray(test_subject)
test_scene_word_array = np.asarray(test_scene_words)
test_scene_sum_array = np.asarray(test_scene_sum)
test_dependent_array = np.asarray(test_dependent)
test_verb_array = np.asarray(test_verb)

np.savetxt('test_subjects.csv', test_subject_array, delimiter=',')
np.savetxt('test_scene_objects.csv', test_scene_word_array, delimiter=',')
np.savetxt('test_scene_sums.csv', test_scene_sum_array, delimiter=',')
np.savetxt('test_dependents.csv', test_dependent_array, delimiter=',')
np.savetxt('test_verbs.csv', test_verb_array, delimiter=',')

# Add Training Data
train_subject = []
train_scene_words = []
train_scene_sum = []
train_dependent = []
train_verb = []

for i in range(2000, len(shuffle_array)):
    x = shuffle_array[i]
    train_subject.append(subject_array[x])
    train_scene_words.append((scene_word_array[x]))
    train_scene_sum.append(scene_sum_array[x])
    train_dependent.append(dependent_array[x])
    train_verb.append(verb_array[x])

train_subject_array = np.asarray(train_subject)
train_scene_word_array = np.asarray(train_scene_words)
train_scene_sum_array = np.asarray(train_scene_sum)
train_dependent_array = np.asarray(train_dependent)
train_verb_array = np.asarray(train_verb)

np.savetxt('train_subjects.csv', train_subject_array, delimiter=',')
np.savetxt('train_scene_objects.csv', train_scene_word_array, delimiter=',')
np.savetxt('train_scene_sums.csv', train_scene_sum_array, delimiter=',')
np.savetxt('train_dependents.csv', train_dependent_array, delimiter=',')
np.savetxt('train_verbs.csv', train_verb_array, delimiter=',')


