import datetime

import pandas as pd
import pytz
import tweepy
from django.db import models
from tqdm import tqdm

from utilities.logging.log import log


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

        log('Collecting tweets')
        tweepy_items = tweepy.Cursor(
            api.user_timeline,
            user_id=userid,
            include_rts=False,
            exclude_replies=True,
            count=100,
            tweet_mode='extended'
        ).items()
        progress = tqdm(tweepy_items, ascii=True)

        for i, tweet in enumerate(progress):
            progress.set_description(f"[INFO]: Processing {count} tweet")
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
