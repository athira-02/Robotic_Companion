import csv
import math
import random
from gensim.test.utils import datapath
from gensim.models import KeyedVectors

# word2vec = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"), binary=True)  # C bin format
word2vec = KeyedVectors.load_word2vec_format(
    datapath("/home/athira/PycharmProjects/Word2VecSketchyScene/word2vec_vectors.bin"), binary=True)  # C bin format

# Edits made to sketchy scene words: street lamp--> lamp and picnic rug--> rug
# Minimum Similarity: -0.05387874     Words: balloon and bench
# Maximum Similarity: 0.76094556      Words: cat and dog
sketchy_scene_words = ['words', 'airplane', 'apple', 'balloon', 'banana', 'basket', 'bee', 'bench', 'bicycle', 'bird',
                       'bottle',
                       'bucket', 'bus', 'butterfly', 'car', 'cat', 'chair', 'chicken', 'cloud', 'cow', 'cup',
                       'dinnerware', 'dog',
                       'duck', 'fence', 'flower', 'grape', 'grass', 'horse', 'house', 'moon', 'mountain', 'people',
                       'rug',
                       'pig', 'rabbit', 'road', 'sheep', 'sofa', 'star', 'lamp', 'sun', 'table', 'tree', 'truck',
                       'umbrella', 'others']

# Finding the Maximum and Minimum Similarity Between Words
'''
#Minimum and Maximum Similarity
min_sim = word2vec.similarity('airplane', 'apple')
max_sim = word2vec.similarity('airplane', 'apple')
min_words = ['airplane', 'apple']
max_words = ['airplane', 'apple']

for i in range(1, len(sketchy_scene_words)):
    for j in range(1, len(sketchy_scene_words)):
        if i != j:
            similarity = word2vec.similarity(sketchy_scene_words[i], sketchy_scene_words[j])
            if similarity < min_sim:
                min_sim = similarity
                min_words = [sketchy_scene_words[i], sketchy_scene_words[j]]
            if similarity > max_sim:
                max_sim = similarity
                max_words = [sketchy_scene_words[i], sketchy_scene_words[j]]

print('Minimum Similarity: ' + str(min_sim) + " Words: " + min_words[0] + " and " + min_words[1])
print('Maximum Similarity: ' + str(max_sim) + " Words: " + max_words[0] + " and " + max_words[1])
'''

perplexity_sum = 0
soft_perplexity_sum = 0

number_of_samples = 535

for sample_number in range(1, number_of_samples + 1):

    # Get the list of words
    scene_list = []
    word_list = []
    with open('../Data/validation_data.csv') as fd:
        reader = csv.reader(fd)
        scene_list = [row for idx, row in enumerate(reader) if idx == sample_number - 1]

    scene_list = scene_list[0]
    del scene_list[0]

    for i in scene_list:
        if int(i) != 46:
            word_list.append(sketchy_scene_words[int(i)])

    # Remove an object randomly
    random_index = random.randint(0, len(word_list)-1)

    removed_word = word_list.pop(random_index)
    removed_word_num = sketchy_scene_words.index(removed_word)

    # Predicting the Word to Add
    max_similarity = 0
    selected_word = ''
    selected_number = 0
    scores = [0]

    for i in range(1, len(sketchy_scene_words)):
        total_similarity = 0
        if sketchy_scene_words[i] not in word_list:
            for word in word_list:
                total_similarity += word2vec.similarity(word, sketchy_scene_words[i])
            # print('Word: ' + sketchy_scene_words[i] + '  Score: ' + str(total_similarity))

        total_similarity = abs(total_similarity)  # In case it is negative
        scores.append(total_similarity)
        if total_similarity > max_similarity:
            max_similarity = total_similarity
            selected_word = sketchy_scene_words[i]
            selected_number = i

    # Normalize Scores
    score_sum = sum(scores)
    normalized_scores = []
    for i in scores:
        x = i / score_sum
        normalized_scores.append(x)

    # Softmax
    soft_score_sum = 0;
    for i in range(1, len(scores)):
        exponent = math.e ** scores[i]
        soft_score_sum += exponent

    soft_scores = [0]
    for i in range(1, len(scores)):
        exponent = (math.e ** scores[i]) / soft_score_sum
        soft_scores.append(exponent)


    # Perplexity
    p = normalized_scores[removed_word_num]
    p_soft = soft_scores[removed_word_num]

    perplexity_sum += math.log(p, 2)
    soft_perplexity_sum += math.log(p_soft, 2)

    print("Sample: " + str(sample_number) + "   Removed Word: " + removed_word + "    Selected Word: " + selected_word +
          "    Probability: " + str(p) + "    Softmax Probability: " + str(p_soft))

    '''
    print('Scene: ')
    print(word_list)
    print('Removed Word: ' + removed_word)
    print("Selected Word: " + selected_word)
    print("Selected Word Score: " + str(max_similarity))
    '''
# Calculate the Perplexity
exponent = (-perplexity_sum/number_of_samples)
soft_exponent = (-soft_perplexity_sum/number_of_samples)
perplexity = 2 ** exponent
soft_perplexity = 2 ** soft_exponent

print("Perplexity: " + str(perplexity))
print("Softmax Perplexity: " + str(soft_perplexity))
