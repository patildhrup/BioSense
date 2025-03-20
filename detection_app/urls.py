from django.urls import path
from .views import detect, index

urlpatterns = [
    path('', index, name='index'),
    path('detect/', detect, name='detect'),
]
