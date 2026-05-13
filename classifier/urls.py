from django.urls import path  # Import path to define URL patterns
from . import views  # Import the views module from the current directory

urlpatterns = [  # List of URL patterns for the classifier app
    path('', views.home, name='home'),  # Route the root URL to the home view
    path('predict', views.predict, name='predict'),  # Route /predict to the predict API view
]
