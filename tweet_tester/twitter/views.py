import json
import os
from operator import itemgetter

import keras
import numpy as np
from cnn_lstm.model.model import CNNLSTMModel
from cnn_lstm.model.preprocessing import (EmbeddingMatrix, PreProcessor,
                                          TweetTokenizer)
from django.shortcuts import redirect, render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utilities.logging.log import log

from .models import CleanTweetModel, Tweepy, TweetModel
from .config.model_parameters import setup_params


def index(req):
    return render(req, 'index.html')


def date_settings(req):
    return render(req, 'date.html')


def search_user(req):
    if req.POST:
        username = req.POST.get('username', 'no_val')

        if username == 'no_val':
            return render(req, 'index.html')

        # EXTRACT TWITTER USER INFO
        user_info = Tweepy.search_user_info(username)
        req.session['user_id'], req.session['user_handle'], req.session['user_auth'] = itemgetter(
            'user_id', 'user_handle', 'user_auth')(user_info)

    return render(req, 'date.html')


def extract_tweets(req):
    if req.POST:
        user_id = req.session['user_id']
        since_date = req.POST.get('since_date')
        end_date = req.POST.get('end_date')

        # EXTRACT USER'S TWEETS w/ TIME FRAME
        tweets = Tweepy.get_tweets(user_id, since_date, end_date)
        tweets_json = json.dumps({'tweets': tweets}, default=str)

        # STORE EXTRACTED TWEETS
        query_record = TweetModel(str(
            req.session.session_key), req.session['user_handle'], since_date, end_date, tweets_json)

        query_record.save()

        return render(req, 'user_tweets.html', {'output': map(lambda t: {'text': t[0], 'timestamp': t[1], 'vectors': [], 'matrix': []}, tweets)})


def analyze_tweets(req):
    # RETRIEVE USER'S TWEETS
    record = TweetModel.objects.get(session_key=req.session.session_key)
    query = {
        'user_handle': record.user_handle,
        'since_date': record.tweets_since_date,
        'end_date': record.tweets_end_date,
        'tweets': json.loads(record.tweets_json)['tweets'],
        'query_date': record.query_date
    }

    if (query['user_handle'] == req.session['user_handle']):
        # CLEAN TWEETS
        cleaned = list(clean_tweets(
            list(map(lambda t: t[0], query['tweets']))
        ))
        # TOKENIZE TWEETS
        vectors = tokenize_tweets(cleaned)
        # INITIALIZE MODEL
        model = CNNLSTMModel(vectors['tokenizer'].tokenizer, vectors['matrix'])
        # CLASSIFY TWEETS
        test_res = model.test(np.array(vectors['vectors']))

        tweets = []

        for i, t in enumerate(cleaned):
            o = {
                'original':  query['tweets'][i][0],
                'cleaned': t,
                'pred_cat': test_res['preds'][i]
            }
            tweets.append(o)

        query_record = CleanTweetModel(
            str(req.session.session_key),
            req.session['user_handle'],
            json.dumps({'tweets': tweets}, default=str),
            test_res['user_pred'][0],
            test_res['user_pred'][1],
            test_res['user_pred'][2]
        )
        query_record.save()

        return redirect('classify_user')

    return redirect('index')


def clean_tweets(tweets):
    log('Cleaning tweets')
    cleaner = PreProcessor()
    pbar = tqdm(tweets)
    clean = []
    i = 0

    for t in pbar:
        clean.append(cleaner.clean_tweet(t))
        pbar.set_description("[INFO]: Cleaning %i tweet" % i)
        i += 1

    log('Cleaning process complete')
    return clean


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


def classify_user(req):
    model_res = CleanTweetModel.objects.get(
        session_key=req.session.session_key)
    pred_res = [model_res.user_anx_class,
                model_res.user_dep_class, model_res.user_nan_class]
    user_pred = CNNLSTMModel.LABELS[np.argmax(pred_res)]

    output = {
        'user_pred': user_pred,
        'anx_pred': '{:.0%}'.format(pred_res[0]),
        'dep_pred': '{:.0%}'.format(pred_res[1]),
        'nan_pred': '{:.0%}'.format(pred_res[2]),
    }

    return render(req, 'classification.html', output)


def list_tweets(req):
    # RETRIEVE TWEETS FROM DB
    model_res = CleanTweetModel.objects.get(
        session_key=req.session.session_key)
    tweet_res = json.loads(model_res.clean_tweets_json)['tweets']

    tweet_out = list(map(lambda x: {
        'tweet': x['original'],
        'clean': x['cleaned'],
        'anx': '{:.0%}'.format(x['pred_cat'][0]),
        'dep': '{:.0%}'.format(x['pred_cat'][1]),
        'nan': '{:.0%}'.format(x['pred_cat'][2])
    }, tweet_res))

    return render(req, 'results.html', {'output': tweet_out})


def train(train=None, val=None, epochs=10):
    log('Testing dat initialized')

    # TOKENIZE TRAINING AND TESTING DATASET
    train_vector = tokenize_tweets(train['x_train'])
    val_vector = tokenize_tweets(val['x_val'])

    # INITIALIZE MODEL
    model = CNNLSTMModel(
        train_vector['tokenizer'].tokenizer,
        train_vector['matrix'],
        setup_params(EmbeddingMatrix, TweetTokenizer)
    )

    # TRAIN MODEL
    result = model.train(
        train_vector['vectors'],
        train['y_train'],
        val_vector['vectors'],
        val['y_val'],
        epochs
    )

    return result


def train_model(req):
    if req.POST:
        epochs = int(req.POST.get('epoch'))
        div_train = req.POST.get('get-from-train')
        train_data = {}

        log('Initializing training data')
        if 'train-data' in req.FILES:
            # PARSE UPLOADED TRAINING DATA
            file = req.FILES['train-data']
            data = file.read()
            train_data = json.loads(data)

            # CLEAN TRAINING DATA TWEETS
            train_data['clean'] = clean_tweets(train_data['x_train'])

            # VECTORIZE LABELS
            encoder = LabelEncoder()
            encoder.fit(train_data['y_train'])
            encoded_y = encoder.transform(train_data['y_train'])
            train_data['y_train'] = keras.utils.np_utils.to_categorical(
                encoded_y)

        log('training data initialized')
        log('Initializing testing data')
        if div_train:
            div_ratio = int(req.POST.get('train-val-div')) / 100
            # SPLIT TRAINING DATA
            x_train, x_val, y_train, y_val = train_test_split(
                train_data['x_train'],
                train_data['y_train'],
                test_size=div_ratio,
                random_state=42
            )

            train_data = {'x_train': x_train, 'y_train': y_train}
            val_data = {'x_val': x_val, 'y_val': y_val}

            return render(req, 'train.html', {'result': train(
                train=train_data,
                val=val_data,
                epochs=epochs
            )})
        else:
            # PARSE UPLOADED TESTING DATA
            if 'val-data' in req.FILES:
                file = req.FILES['val-data']
                data = file.read()
                val_data = json.loads(data)

            encoder = LabelEncoder()
            encoder.fit(val_data['y_val'])
            encoded_y = encoder.transform(val_data['y_val'])
            val_data['y_val'] = keras.utils.np_utils.to_categorical(
                encoded_y)

            return render(req, 'train.html', {'result': train(
                train=train_data,
                val=val_data,
                epochs=epochs
            )})


def setup_train(req):
    return render(req, 'train.html')
