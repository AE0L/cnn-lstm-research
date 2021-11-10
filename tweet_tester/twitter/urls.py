from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('date/', views.date_settings, name="date_settings"),
    path('user_details/', views.search_user, name="search_user"),
    path('user_tweets/', views.extract_tweets, name="extract_tweets"),
    path('analyze/', views.analyze_tweets, name="analyze_tweets"),
    path('classify/', views.classify_user, name="classify_user")
]
