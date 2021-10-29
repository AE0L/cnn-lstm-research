import tweepy
from tweepy import OAuthHandler
import pandas as pd
import datetime
import pytz

#----------------------------function-----------------------------------
class DataExtract:

    def process(userid,sinceDate,endDate):
        youarehappy = "yes"
        access_token = '1449676365893038088-pdugze1ET3LcXHZ4MhgW5KLtDNgi2k'
        access_token_secret = 'bT4xTEVL5IrCkv1MkrorKjzdID4RCPNVankEMmOlALNJL'
        consumer_key = 'fSqFCU8DvkTd8XitEaWNA3ASN'
        consumer_secret = 'fnSlukoXfY4JZtodkqpPZLL8uBg3stk7B9rzQch2tpbgP7ldkT'

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)


        api = tweepy.API(auth)
        tweets = []
        utc=pytz.UTC
        count = 1
        
       
        
        date = datetime.datetime.strptime(str(sinceDate), "%Y-%m-%d")
        startDate = datetime.datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=utc)

        date1 = datetime.datetime.strptime(str(endDate), "%Y-%m-%d")
        lastDate = datetime.datetime(date1.year, date1.month, date1.day, 0, 0, 0, tzinfo=utc)
     
        for tweet in tweepy.Cursor(api.user_timeline, user_id=userid, include_rts = False, exclude_replies=True, count=100, tweet_mode='extended').items():
            if tweet.lang == 'en':
                if tweet.created_at <= lastDate and tweet.created_at >= startDate:
                    try:
                        data = [tweet.created_at, tweet.full_text, tweet.user._json['screen_name'], tweet.user._json['name']]
                        data = tuple(data)

                        tweets.append(data)
                        
                        print(count)
                        count += 1

                    except tweepy.TweepError as e:
                        print(e.reason)
                        continue

                    except StopIteration:
                        break
                else:
                    print ("Finished. Retrieved " + str(count-1) + " tweets.")
                    break

                df = pd.DataFrame(tweets,
                columns=['created_at', 'tweet_text', 'screen_name', 'name'])

                #rawtweet_df = pd.DataFrame(raw_tweets,
                #columns=['tweet_text'])

                df.to_csv(path_or_buf='/Users/joseph/Desktop/extractor_results/trial.csv', index=False)
                #rawtweet_df.to_csv(path_or_buf='/Users/joseph/Desktop/extractor_results/trial_rawtweets.csv', index=False)
                
          

