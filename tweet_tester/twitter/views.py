from django.shortcuts import render
from .models import Tweepy


def index(req):
    return render(req, 'index.html')


def date(req):
    return render(req, 'date.html')


def search_user(req):
    if req.POST:
        username = req.POST.get('username', 'no_val')

        # username empty
        if username == 'no_val':
            return render(req, 'index.html')

        user_info = Tweepy.search_user_info(username)
        req.session['user_id'] = user_info['user_id']

    return render(req, 'date.html', user_info)


def extract_tweets(req):
    tweets = {}

    if req.POST:
        user_id = req.session['user_id']
        since_date = req.POST.get('since_date')
        end_date = req.POST.get('end_date')
        tweets['output'] = Tweepy.get_tweets(user_id, since_date, end_date)

    return render(req, 'user_tweets.html', tweets)
