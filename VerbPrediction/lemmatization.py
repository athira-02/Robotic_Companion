import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# Adjusted Lemmatizer
class AdjustedLemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize(self, word, part_of_speech):
        if part_of_speech == wordnet.VERB and word == 'scared':
            return 'scare'
        elif part_of_speech == wordnet.VERB and word == 'felt':
            return 'feel'
        elif part_of_speech == wordnet.VERB and word == 'feed':
            return 'feed'
        elif part_of_speech == wordnet.VERB and word == 'fell':
            return 'fall'
        else:
            return self.lemmatizer.lemmatize(word, part_of_speech)