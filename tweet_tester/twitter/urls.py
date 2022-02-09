from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('user_details/', views.search_user, name="search_user"),

    path('date/', views.date_settings, name="date_settings"),

    path('user_tweets/', views.extract_tweets, name="extract_tweets"),
    path('check-extract-tweets-process/<str:task_id>/', views.check_extract_tweets_process, name="check_extract_tweets_process"),
    path('view_extract_tweets/', views.view_extract_tweets, name="view_extract_tweets"),

    path('analyze/', views.analyze_tweets, name="analyze_tweets"),
    path('classify/', views.classify_user, name="classify_user"),
    path('list/', views.list_tweets, name="list_tweets"),

    path('train/', views.setup_train, name="setup_train"),
    path('train-model/', views.train_model, name="train_model"),
]
