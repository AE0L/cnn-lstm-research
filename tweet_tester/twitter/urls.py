from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('date/', views.date, name='date'),
    path('user_details/', views.search_user, name="search_user"),
    path('user_tweets/', views.extract_tweets, name="extract_tweets")
]
