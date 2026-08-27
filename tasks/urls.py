from django.urls import path, include
from rest_framework import routers
from tasks import views

router = routers.DefaultRouter()  #muestra el CRUD para usar en la aplicacion 
router.register(r'users', views.UserView, 'users')
router.register(r'diagnosticos', views.DiagnosticoView, 'diagnosticos')
router.register(r'manuals', views.ManualView, 'manuals')



urlpatterns = [
    path("api/v1/", include(router.urls)),
    path("api/v1/validate-credentials/", views.UserView.validateCredentials),
    path("api/v1/validate-image/", views.UserView.evaluateImage),
]