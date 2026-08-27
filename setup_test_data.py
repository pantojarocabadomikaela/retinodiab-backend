import sys, os
sys.path.insert(0, r'D:\Proyectos\1) node\template\proyecto_sde\backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_ia_api.settings')
import django; django.setup()
from tasks.models import User, Diagnostico, Manual
from datetime import datetime

users = [
    {'email': 'admin@hospital.com', 'password': 'admin123', 'nombre': 'Dr. Garcia', 'rol': 'administrador', 'diabetes': False, 'fecha_nacimiento': datetime(1980, 1, 1)},
    {'email': 'maria@ejemplo.com', 'password': 'maria123', 'nombre': 'Maria Lopez', 'rol': 'paciente', 'diabetes': True, 'fecha_nacimiento': datetime(1995, 3, 20)},
    {'email': 'carlos@ejemplo.com', 'password': 'carlos123', 'nombre': 'Carlos Ruiz', 'rol': 'servicio', 'diabetes': False, 'fecha_nacimiento': datetime(1988, 7, 10)},
]
for u in users:
    obj, created = User.objects.get_or_create(email=u['email'], defaults=u)
    print(f"User {u['email']}: {'CREATED' if created else 'EXISTS'} (id={obj.id})")

print(f"Total users: {User.objects.count()}")

# Create sample manuals if empty
if Manual.objects.count() == 0:
    Manual.objects.create(tipo='admin', fuente='<h1>Guia para Administradores</h1><p>Este manual explica como administrar el sistema RETINODiAB.</p>')
    Manual.objects.create(tipo='servicio', fuente='<h1>Guia para Personal de Servicio</h1><p>Este manual explica como registrar diagnosticos.</p>')
    Manual.objects.create(tipo='paciente', fuente='<h1>Guia para Pacientes</h1><p>Este manual explica como consultar sus diagnosticos.</p>')
    print(f"Created {Manual.objects.count()} manuals")
else:
    print(f"Manuals already exist: {Manual.objects.count()}")
