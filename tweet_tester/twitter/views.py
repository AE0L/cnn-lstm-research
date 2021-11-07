import json
import numpy as np
from django.shortcuts import render, redirect
from .models import CNNLSTMModel, EmbeddingMatrix, Tweepy, PreProcessor, TweetModel, TweetTokenizer
from operator import itemgetter
from ast import literal_eval
import os


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

        return render(req, 'user_tweets.html', {'output': map(lambda t: {'text': t[0], 'vectors': [], 'matrix': []}, tweets)})


def analyze_tweets(req):
    record = TweetModel.objects.get(session_key=req.session.session_key)
    query = {
        'user_handle': record.user_handle,
        'since_date': record.tweets_since_date,
        'end_date': record.tweets_end_date,
        'tweets': json.loads(record.tweets_json)['tweets'],
        'query_date': record.query_date
    }

    if (query['user_handle'] == req.session['user_handle']):
        cleaned = list(clean_tweets(
            list(map(lambda t: t[0], query['tweets']))))
        vectors = tokenize_tweets(cleaned)
        output = []

        model = CNNLSTMModel(vectors['tokenizer'].tokenizer, vectors['matrix'])
        model.test(model.model.predict(np.array(vectors['vectors'])))

        for i, t in enumerate(cleaned):
            o = {
                'text': t,
                'vector': vectors['vectors'][i],
                'matrix': vectors['matrix'][i]
            }
            output.append(o)

        return render(req, 'user_tweets.html', {'output': output})
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
