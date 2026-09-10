from django.db import models
from django.contrib.auth.hashers import (
    check_password as django_check_password,
    identify_hasher,
    make_password,
)

# Crea los modelos de datos de los usuarios

class User(models.Model):
    email = models.CharField(max_length = 40, unique = True)
    nombre = models.CharField(max_length = 40)
    # Django recomienda 128 caracteres para almacenar sus hashes de contraseña.
    password = models.CharField(max_length=128)
    rol = models.CharField(max_length = 40)
    diabetes = models.BooleanField(default = False)
    fecha_nacimiento = models.DateTimeField()

    #para mostrar el nombre del usuario en el admin
    def __str__(self):
        return self.nombre

    def set_password(self, raw_password):
        """Convierte una contraseña en texto plano en un hash seguro de Django."""
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return django_check_password(raw_password, self.password)

    def save(self, *args, **kwargs):
        # Protege también las altas hechas fuera de la API (admin, scripts, shell).
        password_was_hashed = False
        try:
            identify_hasher(self.password)
        except ValueError:
            self.set_password(self.password)
            password_was_hashed = True

        # Si save() limita las columnas a actualizar, incluye el nuevo hash.
        if password_was_hashed and kwargs.get('update_fields') is not None:
            kwargs['update_fields'] = set(kwargs['update_fields']) | {'password'}

        super().save(*args, **kwargs)

    @staticmethod
    def authenticate(email, password):
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return None

        return user if user.check_password(password) else None

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
