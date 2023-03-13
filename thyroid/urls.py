from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
    url(r'^$', views.index),#调用刚才写的hello world
    url(r'^roi/', views.showroi),
    url(r'^class/', views.predict_class1),
    url(r'^round/', views.cal_roundness),
    url(r'^at/', views.cal_at),
    url(r'^foci/', views.cal_foci),
    url(r'^index/', views.clean_tmp),
    url(r'^save/', views.save_modify)
]