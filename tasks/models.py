from django.db import models
from django.contrib.auth.hashers import check_password

# Crea los modelos de datos de los usuarios

class User(models.Model):
    email = models.CharField(max_length = 40, unique = True)
    nombre = models.CharField(max_length = 40)
    password = models.CharField(max_length = 40)
    rol = models.CharField(max_length = 40)
    diabetes = models.BooleanField(default = False)
    fecha_nacimiento = models.DateTimeField()

    #para mostrar el nombre del usuario en el admin
    def __str__(self):
        return self.nombre
    
    def authenticate(email, password):
        user = User.objects.get(email = email)
        try:
            if password == user.password:
                return user
            else:
                return None
        except User.DoesNotExist:
            return None

class Diagnostico(models.Model):
    email = models.CharField(max_length=40)
    nombre = models.CharField(max_length=40)
    imagen = models.TextField()
    resultado = models.CharField(max_length=40)
    observaciones = models.CharField(max_length=40)

    def __str__(self):
        return self.nombre

class Manual(models.Model):
    tipo = models.CharField(max_length=40)
    fuente = models.TextField()

    def __str__(self):
        return self.tipo
