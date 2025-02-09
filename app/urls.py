from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chest_CT/', views.chest_CT, name='chest_CT'),
    path('chest_Xray/', views.chest_Xray, name='chest_Xray'),
    path('brain_MRI/', views.brain_MRI, name='brain_MRI'),
    path('api/app/', views.chatbot_response, name='chatbot_response'),
    path('predict_image_Xray/', views.predict_image_Xray, name='predict_image_Xray'),
    path('predict_image_CT/', views.predict_image_CT, name='predict_image_CT'),
    path('predict_image_Brain/', views.predict_image_Brain, name='predict_image_Brain'),
]


from django.conf import settings
from django.conf.urls.static import static

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



