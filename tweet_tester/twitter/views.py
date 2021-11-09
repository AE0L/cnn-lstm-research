from datetime import datetime
import json
import keras
import numpy as np
from django.shortcuts import render, redirect
from pytz import utc
from .models import CNNLSTMModel, EmbeddingMatrix, Tweepy, PreProcessor, TweetModel, TweetTokenizer
from operator import itemgetter
from ast import literal_eval
import os
from sklearn.preprocessing import LabelEncoder


def index(req):
    return render(req, 'index.html')


def date_settings(req):

    return render(req, 'date.html')


def search_user(req):
    if req.POST:
        username = req.POST.get('username', 'no_val')

        # username empty
        if username == 'no_val':
            return render(req, 'index.html')

        user_info = Tweepy.search_user_info(username)
        req.session['user_id'], req.session['user_handle'], req.session['user_auth'] = itemgetter(
            'user_id', 'user_handle', 'user_auth')(user_info)

    return render(req, 'date.html')


def extract_tweets(req):
    if req.POST:
        user_id = req.session['user_id']
        since_date = req.POST.get('since_date')
        end_date = req.POST.get('end_date')
        tweets = Tweepy.get_tweets(user_id, since_date, end_date)
        tweets_json = json.dumps({'tweets': tweets}, default=str)

        query_record = TweetModel(str(
            req.session.session_key), req.session['user_handle'], since_date, end_date, tweets_json)

        query_record.save()

        return render(req, 'user_tweets.html', {'output': map(lambda t: {'text': t[0], 'timestamp': t[1], 'vectors': [], 'matrix': []}, tweets)})


def analyze_tweets(req):
    record = TweetModel.objects.get(session_key=req.session.session_key)
    query = {
        'user_handle': record.user_handle,
        'since_date': record.tweets_since_date,
        'end_date': record.tweets_end_date,
        'tweets': json.loads(record.tweets_json)['tweets'],
        'query_date': record.query_date
    }

    # module_dir = os.path.dirname(__file__)
    # train_dir_path = os.path.join(module_dir, 'res/train_test.json')

    # x_train, y_train = [], []

    # with open(train_dir_path, encoding='utf-8') as json_file:
    #     data = json.load(json_file)
    #     x_train, y_train = data['x_train'], data['y_train']
    
    # train_clean = list(clean_tweets(x_train))
    # train_vector = tokenize_tweets(train_clean)

    # encoder = LabelEncoder()
    # encoder.fit(y_train)
    # encoded_y = encoder.transform(y_train)
    # tmp_y = keras.utils.np_utils.to_categorical(encoded_y)

    if (query['user_handle'] == req.session['user_handle']):
        debug = False
        cleaned = list(clean_tweets(
            list(map(lambda t: t[0], query['tweets']))))
        vectors = tokenize_tweets(cleaned)
        model = CNNLSTMModel(vectors['tokenizer'].tokenizer, vectors['matrix'])
        # model.train(np.array(train_vector['vectors']), np.array(tmp_y))
        test_res = model.test(np.array(vectors['vectors']))

        pred_cats = list(
            map(lambda x: model.LABELS[np.argmax(x)], test_res['preds'])
        )

        if debug == True: print('RESULTS:')

        # DEBUG
        if debug == True:
            for i, t in enumerate(cleaned):
                print('cleaned tweet:', t)
                print('predicted class:', pred_cats[i])
                print()

        tweets = []

        for i, t in enumerate(cleaned):
            o = {
                'original':  query['tweets'][i][0],
                'cleaned': t,
                'pred_cat': pred_cats[i]
            }
            tweets.append(o)

        return render(req, 'results.html', {
            'query': query,
            'tweets': tweets,
            'user_pred': test_res['pred_cat']
        })
    return redirect('index')


def clean_tweets(tweets):
    cleaner = PreProcessor()
    return map(lambda t: cleaner.clean_tweet(t), tweets)


def tokenize_tweets(cleaned_tweets):
    tokenizer = TweetTokenizer(cleaned_tweets)
    tokenizer.train_tokenize()
    vectors = tokenizer.vectorize(cleaned_tweets)

    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, 'res/glove.txt')

    embedding_matrix = EmbeddingMatrix()
    matrix = embedding_matrix.construct_embedding_matrix(
        file_path, tokenizer.tokenizer.word_index)

    return {'vectors': vectors, 'matrix': matrix, 'tokenizer': tokenizer}
