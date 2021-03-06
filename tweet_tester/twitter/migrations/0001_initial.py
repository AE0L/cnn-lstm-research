# Generated by Django 3.2.8 on 2021-10-31 12:49

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='TweetModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('session_key', models.TextField()),
                ('user_handle', models.TextField()),
                ('tweets_since_date', models.DateField()),
                ('tweets_end_date', models.DateField()),
                ('query_date', models.DateField(auto_now=True)),
                ('tweet_file', models.TextField()),
            ],
        ),
    ]
