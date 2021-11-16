import re
import string

import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords, words
from spellchecker.spellchecker import SpellChecker
from tqdm import tqdm


class PreProcessor:
    def __init__(self):
        self.spell_checker = SpellChecker()
        self.stopwords = stopwords.words('english')

    def correct_spelling(self, tweet):
        tweet = tweet.split()
        mispelled = self.spell_checker.unknown(tweet)
        result = map(lambda word: self.spell_checker.correction(
            word) if word in mispelled else word, tweet)

        return " ".join(result)

    def clean_tweet(self, tweet):
        tweet = tweet.lower().strip()
        # remove URLs
        tweet = re.sub(r'http\S+', '', tweet)
        # remove punctuations
        puncts = str.maketrans('', '', string.punctuation)
        tweet = tweet.translate(puncts)
        # remove digits
        digits = str.maketrans('', '', string.digits)
        tweet = tweet.translate(digits)
        # remove emojis
        tweet = tweet.encode('ascii', 'ignore')
        tweet = tweet.decode('UTF-8').strip()
        # check spelling
        tweet = self.correct_spelling(tweet)
        # remove stop words
        tweet = ' '.join([word for word in tweet.split()
                         if word not in self.stopwords])
        # remove non-english words
        eng_words = set(words.words())
        tweet = ' '.join([w for w in wordpunct_tokenize(tweet)
                         if w.lower() in eng_words or not w.isalpha()])
        tweet = ' '.join([w for w in tweet.split() if len(w) > 2])

        return tweet


class TweetTokenizer:
    TOP_K = 2000
    MAX_SEQUENCE_LENGTH = 100

    def __init__(self, tweets):
        self.tweets = tweets
        self.tokenizer = Tokenizer(num_words=self.TOP_K)

    def train_tokenize(self):
        max_length = len(max(self.tweets, key=len))
        self.max_length = min(max_length, self.MAX_SEQUENCE_LENGTH)
        self.tokenizer.fit_on_texts(self.tweets)

    def vectorize(self, tweets):
        tweets = self.tokenizer.texts_to_sequences(tweets)
        tweets = sequence.pad_sequences(
            tweets, maxlen=self.max_length,
            truncating='post', padding='post'
        )

        return tweets


class EmbeddingMatrix:
    EMBEDDING_VECTOR_LENGTH = 50

    def construct_embedding_matrix(self, glove_file, word_index):
        embedding_dict = {}

        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]

                if word in word_index.keys():
                    vector = np.asarray(values[1:], 'float32')
                    embedding_dict[word] = vector

        num_words = len(word_index) + 1
        embedding_matrix = np.zeros((num_words, self.EMBEDDING_VECTOR_LENGTH))

        pbar = tqdm(word_index.items())

        for word, i in pbar:
            pbar.set_description("[INFO]: Processing %i tweet" % i)
            if i < num_words:
                vect = embedding_dict.get(word, [])

                if len(vect) > 0:
                    embedding_matrix[i] = vect[:self.EMBEDDING_VECTOR_LENGTH]

        return embedding_matrix
