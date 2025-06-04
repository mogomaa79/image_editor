from django.urls import path
from . import views

app_name = 'image_processor'

urlpatterns = [
    path('', views.home, name='home'),
    path('process/', views.process_image_api, name='process_image'),
    path('download/<int:record_id>/', views.download_image, name='download_image'),
    path('modes/', views.get_processing_modes, name='get_modes'),
] 