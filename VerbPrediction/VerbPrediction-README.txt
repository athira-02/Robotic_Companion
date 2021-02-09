Uses data from the Abstract Scene Dataset which contains scenes and sentence desscriptions of what is going on in the scenes. The sentences are parsed to get "subject", "dependent", and "relation" (or verb) for each scene.

dataPreperation.py - run to create input data for each model

verbPrediction.py - multilayer model with 3 hidden layers of 512 nodes and 0.5 Dropout between each layer. trains the model to select from 69 possible verbs.  The input is a 600 dimension vector which is the subject vector concatenated with the 'scene vector' which is the sum of the vectors for each object on the scene. word2vec[subject] + sum_i(word2vec[scene_object_i])

verbPredictionWithAttention.py - model with an attention layer to select from 69 possible verbs. The query is the subject vector and the value is a 6x300 matrix where each row is the vector for a particular object in the scene.

dependentPrediction.py - model to predict the dependent given two inputs: the subject vector and the vectors for all the objects in the scene. The model selects one of the objects in the scene or "none" (i.e. the dependent is not an object on the screen) to be the dependent.

verbPredictionFromDependent.py - model to predict the verb taking in the subject and dependent as input. Tries to predict the relationship between the subject and dependent. Input is subject vector concatentated with dependent vector. There are three hidden dense layers with 512 nodes each and 0.5 Dropout between each layer
