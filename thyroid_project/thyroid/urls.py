from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
    #url(r'^$', views.index),
    url(r'^roi/', views.showroi),
    url(r'^class/', views.predict_class1),
    url(r'^round/', views.cal_roundness),
    url(r'^at/', views.cal_at),
    url(r'^foci/', views.cal_foci),
    url(r'^index/', views.clean_tmp),
    url(r'^save/', views.save_modify),
    path('', views.login_view, name='login_view'),
    path('reset_pwd/', views.reset_view),
    path('reset_pwd1/', views.reset_pwd, name='reset_pwd1'),
    path('index1/', views.login_view_submit),
    path('register/', views.register_view),
    path('registergoto/', views.register_view_submit),
]