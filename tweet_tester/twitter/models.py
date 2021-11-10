import datetime
import os
import re
import string
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import pytz
import tweepy
from django.db import models
from keras import models as kmodels
from keras.layers import Embedding
from nltk.corpus import stopwords, words
from spellchecker.spellchecker import SpellChecker
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.backend import constant, dtype
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.layers.pooling import (GlobalMaxPool1D, MaxPool1D,
                                                    MaxPooling1D)
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.wrappers import TimeDistributed
from tqdm import tqdm

from .config.model_parameters import setup_params
from .utilities.log import LogLevel, log
from .utilities.tqdm_predict_callback import TQDMPredictCallback


class Tweepy:
    @staticmethod
    def search_user_info(username):
        log('Initializing Tweepy')
        access_token = '1449676365893038088-pdugze1ET3LcXHZ4MhgW5KLtDNgi2k'
        access_token_secret = 'bT4xTEVL5IrCkv1MkrorKjzdID4RCPNVankEMmOlALNJL'
        consumer_key = 'fSqFCU8DvkTd8XitEaWNA3ASN'
        consumer_secret = 'fnSlukoXfY4JZtodkqpPZLL8uBg3stk7B9rzQch2tpbgP7ldkT'

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        api = tweepy.API(auth)
        log('Tweepy initilized')

        log('Getting user information')
        username = api.get_user(screen_name=username)
        log('User information obtained')
        screen_name = username.screen_name
        userid = username.id
        acc_auth = 'Public' if username.protected == False else 'Private'

        return {
            'user_handle': screen_name,
            'user_id': userid,
            'user_auth': acc_auth
        }

    @staticmethod
    def get_tweets(userid, since_date, end_date):
        log('Initializing Tweepy')
        access_token = '1449676365893038088-pdugze1ET3LcXHZ4MhgW5KLtDNgi2k'
        access_token_secret = 'bT4xTEVL5IrCkv1MkrorKjzdID4RCPNVankEMmOlALNJL'
        consumer_key = 'fSqFCU8DvkTd8XitEaWNA3ASN'
        consumer_secret = 'fnSlukoXfY4JZtodkqpPZLL8uBg3stk7B9rzQch2tpbgP7ldkT'

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        api = tweepy.API(auth)
        tweets = []
        utc = pytz.UTC
        count = 1
        log('Tweepy initialized')

        date = datetime.datetime.strptime(str(since_date), "%Y-%m-%d")
        start_date = datetime.datetime(
            date.year, date.month, date.day, 0, 0, 0, tzinfo=utc)

        date1 = datetime.datetime.strptime(str(end_date), "%Y-%m-%d")
        last_date = datetime.datetime(
            date1.year, date1.month, date1.day, 0, 0, 0, tzinfo=utc)

        log('Collecting tweets:')
        progress = tqdm(
            tweepy.Cursor(api.user_timeline, user_id=userid, include_rts=False, exclude_replies=True, count=100, tweet_mode='extended').items(), ascii=True
        )

        for tweet in progress:
            progress.set_description("Processing %i tweet" % count)
            if tweet.created_at >= start_date:
                if tweet.created_at <= last_date:
                    try:
                        data = [tweet.created_at, tweet.full_text,
                                tweet.user._json['screen_name'], tweet.user._json['name']]
                        data = tuple(data)

                        tweets.append(data)

                        count += 1

                    except tweepy.TweepError as e:
                        print(e)
                        continue
                    except StopIteration:
                        break
            else:
                break

        log('Tweets successfully collected')
        df = pd.DataFrame(
            tweets, columns=['created_at', 'tweet_text', 'screen_name', 'name'])
        tweets = []

        for index, row in df.iterrows():
            tweets.append([row['tweet_text'], row['created_at']])

        return tweets


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
        tweet = ' '.join([w for w in nltk.wordpunct_tokenize(tweet)
                         if w.lower() in eng_words or not w.isalpha()])
        tweet = ' '.join([w for w in tweet.split() if len(w) > 2])

        return tweet


class TweetTokenizer:
    TOP_K = 2000
    MAX_SEQUENCE_LENGTH = 50

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
    EMBEDDING_VECTOR_LENGTH = 25

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


class CNNLSTMModel:
    LABELS = ['Anxiety', 'Depression', 'None']

    def __init__(self, tokenizer, embedding_matrix):
        log('Initializing the model')
        module_dir = os.path.dirname(__file__)
        self.model_file_path = os.path.join(module_dir, 'res/cnn-lstm-model')

        if os.path.exists(self.model_file_path):
            print('LOADING MODEL...')
            self.model = kmodels.load_model(self.model_file_path)
        else:
            PARAMS = setup_params(EmbeddingMatrix, TweetTokenizer)

            # Embedding Layer
            input_dim = len(tokenizer.word_index) + 1
            output_dim = PARAMS['embedding']['OUTPUT_DIM']
            input_length = PARAMS['embedding']['INPUT_LENGTH']
            weights = [embedding_matrix]
            trainable = PARAMS['embedding']['TRAINABLE']
            # CNN
            filters = PARAMS['cnn']['FILTERS']
            kernel = PARAMS['cnn']['KERNEL']
            cnn_act = PARAMS['cnn']['ACTIVATION']
            # Max Pooling
            pool_size = PARAMS['max-pooling']['POOL_SIZE']
            # LSTM
            lstm_units = PARAMS['lstm']['UNITS']
            dropout = PARAMS['lstm']['DROPOUT']
            # Dense
            dense_units = PARAMS['dense']['UNITS']
            dense_act = PARAMS['dense']['ACTIVATION']
            # Compile
            loss = PARAMS['compile']['LOSS']
            optimizer = PARAMS['compile']['OPTIMIZER']
            metrics = PARAMS['compile']['METRICS']

            embedding_layer = Embedding(
                input_dim,
                output_dim,
                input_length=input_length,
                weights=weights,
                trainable=trainable
            )
            model = Sequential()
            model.add(embedding_layer)
            model.add(Conv1D(filters, kernel, activation=cnn_act))
            model.add(MaxPool1D(pool_size))
            model.add(LSTM(lstm_units, dropout=dropout))
            model.add(Dense(dense_units, activation=dense_act))
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

            self.model = model
            self.model.save(self.model_file_path)
            print(model.summary())
        log('Model successfully initialized')

    def train(self, x_train, y_train):
        epochs = 10
        verbose = 2
        log('Training the model')
        self.model.fit(x_train, y_train, epochs=epochs, verbose=verbose)
        self.model.save(self.model_file_path)
        log('Model training completed')

    def test(self, test_seq):
        log('Classifying tweets')
        preds = self.model.predict(test_seq)
        log('Getting average prediction')
        pred_avg = np.average(preds, axis=0)
        log('Tweets classification completed')

        return {'user_pred': pred_avg, 'preds': preds.tolist()}


class TweetModel(models.Model):
    session_key = models.TextField(primary_key=True)
    user_handle = models.TextField()
    tweets_since_date = models.DateField()
    tweets_end_date = models.DateField()
    tweets_json = models.JSONField()
    query_date = models.DateField(auto_now=True)


class CleanTweetModel(models.Model):
    session_key = models.TextField(primary_key=True)
    user_handle = models.TextField()
    clean_tweets_json = models.JSONField()
    user_anx_class = models.FloatField()
    user_dep_class = models.FloatField()
    user_nan_class = models.FloatField()
