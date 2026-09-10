"""Configuración aislada para ejecutar benchmarks sin tocar la base principal."""

import os

from .settings import *  # noqa: F403


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.environ.get('BENCHMARK_DB_PATH', '/tmp/retinodiab_benchmark.sqlite3'),
    }
}
