from django.urls import path
from .views import detect, index, landing, privacy

urlpatterns = [
    path('', landing, name='landing'),
    path('app/', index, name='index'),
    path('detect/', detect, name='detect'),
    path('privacy/', privacy, name='privacy'),
]
