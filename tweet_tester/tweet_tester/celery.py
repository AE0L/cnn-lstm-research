import os

from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tweet_tester.settings')
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')

app = Celery('tweet_tester')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')