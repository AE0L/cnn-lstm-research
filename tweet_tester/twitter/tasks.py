import json
import os
from tracemalloc import start

import numpy as np
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from cnn_lstm.model.model import CNNLSTMModel
from cnn_lstm.model.preprocessing import (EmbeddingMatrix, PreProcessor,
                                          TweetTokenizer)
from tqdm import tqdm
from utilities.logging.log import log, LogLevel

from .models import CleanTweetModel, Tweepy, TweetModel


@shared_task(bind=True)
def get_user_tweets(self, ses_key, handle, user_id, since_date, end_date):
    # EXTRACT USER'S TWEETS w/ TIME FRAME
    tweets = Tweepy.get_tweets(user_id, since_date, end_date)
    print('QWERTY: ', len(tweets))
    tweets_json = json.dumps({'tweets': tweets}, default=str)

    # STORE EXTRACTED TWEETS
    query_record = TweetModel(
        ses_key,
        handle,
        since_date,
        end_date,
        tweets_json
    )

    query_record.save()

    return 'Done'


@shared_task(bind=True)
def classify_tweets(self, ses_key, user_handle, user_tweets):
    user_tweets = list(map(lambda t: t[0], user_tweets))
    max_len = len(user_tweets) + 3
    pr = ProgressRecorder(self)
    pr.set_progress(0, max_len, 'Cleaning Tweets')
    # CLEAN TWEETS
    cleaned = list(clean_tweets(user_tweets, task=True, pr=pr))
    # TOKENIZE TWEETS
    pr.set_progress(len(cleaned) + 1, max_len, 'Tokenizing tweets')
    vectors = tokenize_tweets(cleaned)
    # CLASSIFY TWEETS
    pr.set_progress(len(cleaned) + 2, max_len, 'Classifying tweets')
    model = CNNLSTMModel(vectors['tokenizer'].tokenizer, vectors['matrix'])
    test_res = model.test(np.array(vectors['vectors']))

    tweets = []

    for i, t in enumerate(cleaned):
        o = {
            'original':  user_tweets[i],
            'cleaned': t,
            'pred_cat': test_res['preds'][i]
        }
        tweets.append(o)

    query_record = CleanTweetModel(
        ses_key,
        user_handle,
        json.dumps({'tweets': tweets}, default=str),
        test_res['user_pred'][0],
        test_res['user_pred'][1],
        test_res['user_pred'][2]
    )

    query_record.save()

    return 'Done'


def clean_tweets(tweets, vector=None, task=False, pr=None):
    log('Cleaning tweets')

    max_len = len(tweets) + 2
    cleaner = PreProcessor()
    clean = []
    rem_vect = []
    pbar = tqdm(enumerate(tweets))

    for i, t in pbar:
        if task == True:
            pr.set_progress(i + 1, max_len, f'Cleaning {i} out of {max_len}')

        pbar.set_description(f'Cleaning {i + 1} tweet out of {max_len}')
        clean_t = cleaner.clean_tweet(t)

        if (len(clean_t) > 0 and clean_t != ''):
                clean.append(clean_t)
        else:
            rem_vect.append(i)

    log(f'removed {len(rem_vect)} empty tweets')
        
    r_vector = [x for i, x in enumerate(vector) if i not in rem_vect]

    log(len(clean))
    log(len(r_vector))

    log('Cleaning process complete')

    return (clean, r_vector)


def tokenize_tweets(cleaned_tweets):
    log('Tokenization process started')
    tokenizer = TweetTokenizer(cleaned_tweets)
    tokenizer.train_tokenize()
    vectors = tokenizer.vectorize(cleaned_tweets)

    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, 'res/glove.txt')

    embedding_matrix = EmbeddingMatrix()
    matrix = embedding_matrix.construct_embedding_matrix(
        file_path, tokenizer.tokenizer.word_index)
    log('Tokenization process completed')

    return {'vectors': vectors, 'matrix': matrix, 'tokenizer': tokenizer}
