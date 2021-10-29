from django.shortcuts import render
from .models import Tweepy


def index(req):
    return render(req, 'index.html')


def date(req):
    return render(req, 'date.html')


def search_user(req):
    if req.POST:
        username = req.POST.get('username', 'no_val')
        user_info = Tweepy.search_user_info(username)

    return render(req, 'date.html', user_info)
