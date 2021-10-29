from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('date/', views.date, name='date'),
    path('user_details/', views.search_user, name="search_user")
]
