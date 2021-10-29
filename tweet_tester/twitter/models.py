from django.db import models
import tweepy


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

        # userDetails(screen_name, userid, accAuth)
        return {
            'user_handle': screen_name,
            'user_id': userid,
            'user_auth': acc_auth
        }
