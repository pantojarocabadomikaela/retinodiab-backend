@echo off
cd /d "D:\Proyectos\1) node\template\proyecto_sde\backend"
call venv\Scripts\activate.bat
python manage.py runserver 8000
