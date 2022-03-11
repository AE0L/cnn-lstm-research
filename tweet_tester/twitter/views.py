import json
import os
from operator import itemgetter
from celery_progress.backend import ProgressRecorder
from django.http.response import JsonResponse

import keras
import numpy as np
from cnn_lstm.model.model import CNNLSTMModel
from cnn_lstm.model.preprocessing import (EmbeddingMatrix, PreProcessor,
                                          TweetTokenizer)
from keras.utils.np_utils import to_categorical
from django.shortcuts import redirect, render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from utilities.logging.log import log, LogLevel

from celery.result import AsyncResult

from .models import CleanTweetModel, Tweepy, TweetModel
from .config.model_parameters import setup_params
from .tasks import get_user_tweets, classify_tweets, clean_tweets, tokenize_tweets


def index(req):
    return render(req, 'index.html')


def date_settings(req):
    return render(req, 'date.html')


# =============================================================================
# QUERYING USER's ACCOUNT INFO
# =============================================================================
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


# =============================================================================
# EXTRACTING USER's TWEETS
# =============================================================================
def extract_tweets(req):
    if req.POST:
        ses_key = str(req.session.session_key)
        handle = req.session['user_handle']
        user_id = req.session['user_id']
        since_date = req.POST.get('since_date')
        end_date = req.POST.get('end_date')

        task = get_user_tweets.delay(
            ses_key, handle, user_id, since_date, end_date)

        return render(req, 'load_user_tweets.html', {'task_id': task.task_id})


def check_extract_tweets_process(req, task_id):
    # task_id = req.GET.get('task_id')
    if task_id:
        async_result = AsyncResult(task_id)
        return JsonResponse({'finish': async_result.ready()})
    return JsonResponse({'finish': False})


def view_extract_tweets(req):
    ses_key = req.session.session_key
    record = TweetModel.objects.get(session_key=ses_key)
    tweets = json.loads(record.tweets_json)['tweets']

    return render(req, 'user_tweets.html', {
        'output': list(map(lambda t: {
            'text': t[0],
            'timestamp': t[1],
        }, tweets))
    })


# =============================================================================
# ANALYZING USER's TWEETS
# =============================================================================
def analyze_tweets(req):
    ses_key = req.session.session_key
    record = TweetModel.objects.get(session_key=ses_key)
    user_handle = record.user_handle
    user_tweets = json.loads(record.tweets_json)['tweets']

    if (user_handle == req.session['user_handle']):
        task = classify_tweets.delay(ses_key, user_handle, user_tweets)
        return render(req, 'load_analyze_tweets.html', {'task_id': task.task_id})

    return redirect('index')


def classify_user(req):
    model_res = CleanTweetModel.objects.get(
        session_key=req.session.session_key)
    pred_res = [model_res.user_anx_class,
                model_res.user_dep_class, model_res.user_nan_class]
    # user_pred = CNNLSTMModel.LABELS[np.argmax(pred_res)]
    user_pred = []

    anx_thresh = 0.88
    dep_thresh = 0.63
    nan_thresh = 0.02

    for i, pred_class in enumerate(pred_res):
        if i == 0 and pred_class > anx_thresh:
            user_pred.append(CNNLSTMModel.LABELS[0])

        if i == 1 and pred_class > dep_thresh:
            user_pred.append(CNNLSTMModel.LABELS[1])

    user_pred = ', '.join(user_pred)

    if len(user_pred) == 0:
        user_pred = 'NONE'

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


# =============================================================================
# TRAINING THE CNN-LSTM MODEL
# =============================================================================
def train(train=None, val=None, epochs=10):
    log('Testing data initialized')

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
            clean_x, clean_y = clean_tweets(
                train_data['x_train'], vector=train_data['y_train'])

            len_anx = 0
            len_dep = 0
            len_nan = 0

            for y in clean_y:
                if 0 in y:
                    len_anx += 1
                if 1 in y:
                    len_dep += 1
                if 2 in y:
                    len_nan += 1

            log(f'ANXIETY class: {len_anx}')
            log(f'DEPRESSION class: {len_dep}')
            log(f'NONE class: {len_nan}')

            # VECTORIZE LABELS
            encoder = MultiLabelBinarizer()
            encoded_y = encoder.fit_transform(clean_y)
            clean_y = np.asarray(encoded_y)

            log('training data initialized')
            log('Initializing testing data')

            if div_train:
                div_ratio = int(req.POST.get('train-val-div')) / 100

                # SPLIT TRAINING DATA
                x_train, x_val, y_train, y_val = train_test_split(
                    clean_x,
                    clean_y,
                    test_size=div_ratio,
                    random_state=42,
                    stratify=clean_y
                )

                log(f'train dataset: {len(x_train)}')
                log(f'test dataset: {len(x_val)}')

                train_data = {'x_train': x_train, 'y_train': y_train}
                val_data = {'x_val': x_val, 'y_val': y_val}

                return render(req, 'train.html', {'result': train(
                    train=train_data,
                    val=val_data,
                    epochs=epochs
                )})


def setup_train(req):
    return render(req, 'train.html')
