from pathlib import Path
import tweepy
from tweepy import OAuthHandler
import pandas as pd
import datetime
import pytz
from time import sleep
from tqdm import tqdm

# ----------------------------function-----------------------------------


class DataExtract:

    def process(userid, sinceDate, endDate):
        youarehappy = "yes"
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

        #startDate_input = input("Enter start date(mm-mm-dd): ")
        date = datetime.datetime.strptime(str(sinceDate), "%Y-%m-%d")
        startDate = datetime.datetime(
            date.year, date.month, date.day, 0, 0, 0, tzinfo=utc)

        date1 = datetime.datetime.strptime(str(endDate), "%Y-%m-%d")
        date1 = date1 + datetime.timedelta(days=1)
        lastDate = datetime.datetime(
            date1.year, date1.month, date1.day, 0, 0, 0, tzinfo=utc)

        print(userid)

        progress = tqdm(
            tweepy.Cursor(api.user_timeline, user_id=userid, include_rts=False, exclude_replies=True, count=100, tweet_mode='extended').items(), ascii=True, unit='tweets'
        )

        for tweet in progress:
            progress.set_description("Processing %i tweets" % count)
                if tweet.lang == 'en':
                    if tweet.created_at >= startDate:
                        if tweet.created_at <= lastDate:
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

        

        output_file = 'trial.csv'
        output_dir = Path('output/')
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_dir / output_file, index=False)
        print('test')
