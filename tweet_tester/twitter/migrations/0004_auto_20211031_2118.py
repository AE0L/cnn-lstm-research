# Generated by Django 3.2.8 on 2021-10-31 13:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('twitter', '0003_rename_tweet_file_tweetmodel_tweets_array'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='tweetmodel',
            name='tweets_array',
        ),
        migrations.AddField(
            model_name='tweetmodel',
            name='tweets_json',
            field=models.JSONField(default={}),
            preserve_default=False,
        ),
    ]
