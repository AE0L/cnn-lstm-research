from django.db import models
from tqdm import tqdm
from pathlib import Path
import tweepy
import pytz
import datetime
import pandas as pd


class Tweepy:
    @staticmethod
    def search_user_info(username):
        access_token = '1449676365893038088-pdugze1ET3LcXHZ4MhgW5KLtDNgi2k'
        access_token_secret = 'bT4xTEVL5IrCkv1MkrorKjzdID4RCPNVankEMmOlALNJL'
        consumer_key = 'fSqFCU8DvkTd8XitEaWNA3ASN'
        consumer_secret = 'fnSlukoXfY4JZtodkqpPZLL8uBg3stk7B9rzQch2tpbgP7ldkT'

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        api = tweepy.API(auth)

        username = api.get_user(screen_name=username)
        screen_name = username.screen_name
        userid = username.id
        acc_auth = 'Public' if username.protected == False else 'Private'

        return {
            'user_handle': screen_name,
            'user_id': userid,
            'user_auth': acc_auth
        }

    @staticmethod
    def get_tweets(userid, since_date):
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

        date = datetime.datetime.strptime(str(since_date), "%Y-%m-%d")
        start_date = datetime.datetime(
            date.year, date.month, date.day, 0, 0, 0, tzinfo=utc)

        print(userid)

        progress = tqdm(
            tweepy.Cursor(api.user_timeline, user_id=userid, include_rts=False, exclude_replies=True, count=100, tweet_mode='extended').items(), ascii=True, unit='tweets'
        )

        for tweet in progress:
            progress.set_description("Processing %i tweets" % count)
            if tweet.created_at >= start_date:
                try:
                    data = [tweet.created_at, tweet.full_text,
                            tweet.user._json['screen_name'], tweet.user._json['name']]
                    data = tuple(data)

                    tweets.append(data)

                    count += 1

                except tweepy.TweepError as e:
                    print(e.reason)
                    continue

                except StopIteration:
                    break
            else:
                break

        print("Finished. Retrieved " + str(count-1) + " tweets.")
        df = pd.DataFrame(tweets,
                          columns=['created_at', 'tweet_text', 'screen_name', 'name'])

        # rawtweet_df = pd.DataFrame(raw_tweets,
        # columns=['tweet_text'])

        output_file = 'trial.csv'
        output_dir = Path('output/')
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_dir / output_file, index=False)
