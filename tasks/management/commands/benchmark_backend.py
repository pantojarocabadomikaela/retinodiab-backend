import base64
import json
import math
import platform
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import django
import torch
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import BaseCommand, call_command
from django.test import Client

from tasks.ai import model_loader
from tasks.ai.config import MODEL_ID
from tasks.ai.preprocessing import decode_rgb_image, preprocess_image_bytes
from tasks.models import Diagnostico, User


class Command(BaseCommand):
    help = 'Mide el rendimiento de las operaciones principales del backend.'

    def add_arguments(self, parser):
        parser.add_argument('--iterations', type=int, default=5)
        parser.add_argument('--warmup', type=int, default=1)
        parser.add_argument('--image', type=str)
        parser.add_argument('--output', type=str, default='performance_results.json')
        parser.add_argument(
            '--prepare-database',
            action='store_true',
            help='Aplica migraciones antes de medir (recomendado con benchmark_settings).',
        )

    def handle(self, *args, **options):
        iterations = options['iterations']
        warmup = options['warmup']
        if iterations < 1 or warmup < 0:
            raise ValueError('iterations debe ser >= 1 y warmup debe ser >= 0.')

        if options['prepare_database']:
            call_command('migrate', interactive=False, verbosity=0)

        image_path = self._resolve_image_path(options['image'])
        image_bytes = image_path.read_bytes()
        run_id = uuid.uuid4().hex[:12]
        email_prefix = f'benchmark-{run_id}'
        client = Client()
        created_user_ids = []

        try:
            auth_user = User.objects.create(
                email=f'{email_prefix}-auth@example.com',
                nombre='Usuario Benchmark',
                password='benchmark-password-2026',
                rol='paciente',
                diabetes=False,
                fecha_nacimiento='2000-01-01T00:00:00Z',
            )
            created_user_ids.append(auth_user.id)

            results = []
            results.append(self._measure(
                'Creación de usuario',
                'Base de datos y hash de contraseña',
                lambda index: self._create_user(email_prefix, index, created_user_ids),
                iterations,
            ))
            results.append(self._measure(
                'Autenticación',
                'Consulta de usuario y verificación del hash',
                lambda _index: User.authenticate(
                    auth_user.email,
                    'benchmark-password-2026',
                ),
                iterations,
                validate=lambda user: user is not None,
            ))
            results.append(self._measure(
                'Decodificación de imagen',
                'Bytes de imagen a matriz RGB',
                lambda _index: decode_rgb_image(image_bytes),
                iterations,
            ))
            results.append(self._measure(
                'Preprocesamiento',
                'Canal verde, CLAHE, resize y normalización',
                lambda _index: preprocess_image_bytes(image_bytes),
                iterations,
            ))

            model_loader._bundle = None
            model_load_start = time.perf_counter_ns()
            bundle = model_loader.get_model_bundle()
            self._synchronize(bundle.device)
            model_load_ms = (time.perf_counter_ns() - model_load_start) / 1_000_000
            results.append(self._single_measurement(
                'Carga inicial del modelo',
                'Lectura del checkpoint y construcción de EfficientNetB0',
                model_load_ms,
            ))

            input_tensor = preprocess_image_bytes(image_bytes)
            for _ in range(warmup):
                self._run_model(bundle, input_tensor)

            results.append(self._measure(
                'Inferencia del modelo',
                'Forward pass y Softmax con el modelo ya cargado',
                lambda _index: self._run_model(bundle, input_tensor),
                iterations,
                before=lambda: self._synchronize(bundle.device),
                after=lambda: self._synchronize(bundle.device),
            ))
            results.append(self._measure(
                'Pipeline IA completo',
                'Preprocesamiento, inferencia y construcción de la clasificación',
                lambda _index: self._predict_image(image_bytes),
                iterations,
                before=lambda: self._synchronize(bundle.device),
                after=lambda: self._synchronize(bundle.device),
            ))

            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            results.append(self._measure(
                'Almacenamiento de diagnóstico',
                'Codificación Base64 ya realizada y escritura en SQLite',
                lambda index: Diagnostico.objects.create(
                    email=auth_user.email,
                    nombre=f'benchmark_{run_id}_{index}',
                    resultado='Moderate',
                    observaciones='benchmark',
                    imagen=encoded_image,
                ),
                iterations,
            ))
            results.append(self._measure(
                'Consulta del historial',
                'GET /api/v1/diagnosticos/ y serialización de resultados',
                lambda _index: client.get('/api/v1/diagnosticos/'),
                iterations,
                validate=lambda response: response.status_code == 200,
            ))
            results.append(self._measure(
                'Respuesta completa del diagnóstico',
                'POST, espera configurada, IA, almacenamiento y respuesta JSON',
                lambda index: self._post_diagnosis(
                    client,
                    auth_user.email,
                    image_bytes,
                    index,
                ),
                iterations,
                validate=lambda response: response.status_code == 200,
            ))

            payload = {
                'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'iterations': iterations,
                'warmup_iterations': warmup,
                'image_path': str(image_path),
                'image_size_bytes': len(image_bytes),
                'model_id': MODEL_ID,
                'device': str(bundle.device),
                'python_version': platform.python_version(),
                'django_version': django.get_version(),
                'torch_version': torch.__version__,
                'platform': platform.platform(),
                'notes': [
                    'Tiempos medidos con time.perf_counter_ns y expresados en milisegundos.',
                    'La carga inicial del modelo se mide una sola vez.',
                    'La respuesta completa incluye el time.sleep(5) configurado en evaluateImage.',
                    'La base de datos del benchmark es independiente de db.sqlite3.',
                ],
                'results': results,
            }
            output_path = Path(options['output']).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            self.stdout.write(self.style.SUCCESS(f'Resultados guardados en {output_path}'))
            for result in results:
                self.stdout.write(
                    f"{result['operation']}: promedio={result['mean_ms']:.3f} ms, "
                    f"p95={result['p95_ms']:.3f} ms"
                )
        finally:
            Diagnostico.objects.filter(email__startswith=email_prefix).delete()
            User.objects.filter(email__startswith=email_prefix).delete()

    @staticmethod
    def _resolve_image_path(value):
        if value:
            path = Path(value).resolve()
        else:
            test_images = Path.cwd() / 'test_images'
            candidates = sorted(test_images.glob('*'))
            path = next((item for item in candidates if item.is_file()), None)
        if path is None or not path.is_file():
            raise FileNotFoundError('No se encontró una imagen para ejecutar el benchmark.')
        return path

    @staticmethod
    def _create_user(prefix, index, created_user_ids):
        user = User.objects.create(
            email=f'{prefix}-{index}@example.com',
            nombre='Usuario Benchmark',
            password='benchmark-password-2026',
            rol='paciente',
            diabetes=False,
            fecha_nacimiento='2000-01-01T00:00:00Z',
        )
        created_user_ids.append(user.id)
        return user

    @staticmethod
    def _synchronize(device):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    @classmethod
    def _run_model(cls, bundle, input_tensor):
        tensor = input_tensor.to(
            device=bundle.device,
            dtype=torch.float32,
            non_blocking=False,
        )
        with torch.inference_mode():
            logits = bundle.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = int(torch.argmax(probabilities, dim=1).item())
        return predicted_class

    @staticmethod
    def _predict_image(image_bytes):
        from tasks.ai.inference import predict_image

        return predict_image(image_bytes)

    @staticmethod
    def _post_diagnosis(client, email, image_bytes, index):
        image = SimpleUploadedFile(
            f'benchmark-{index}.png',
            image_bytes,
            content_type='image/png',
        )
        return client.post(
            '/api/v1/validate-image/',
            {
                'email': email,
                'observaciones': f'benchmark-{index}',
                'imageFile': image,
            },
        )

    @staticmethod
    def _measure(
        operation,
        description,
        function,
        iterations,
        validate=None,
        before=None,
        after=None,
    ):
        samples = []
        for index in range(iterations):
            if before:
                before()
            start = time.perf_counter_ns()
            result = function(index)
            if after:
                after()
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            if validate and not validate(result):
                raise RuntimeError(f'Validación fallida durante: {operation}')
            samples.append(elapsed_ms)
        return Command._summarize(operation, description, samples)

    @staticmethod
    def _single_measurement(operation, description, value_ms):
        return Command._summarize(operation, description, [value_ms])

    @staticmethod
    def _summarize(operation, description, samples):
        ordered = sorted(samples)
        p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
        return {
            'operation': operation,
            'description': description,
            'samples_ms': [round(value, 6) for value in samples],
            'iterations': len(samples),
            'mean_ms': round(statistics.mean(samples), 6),
            'median_ms': round(statistics.median(samples), 6),
            'min_ms': round(min(samples), 6),
            'max_ms': round(max(samples), 6),
            'p95_ms': round(ordered[p95_index], 6),
            'stdev_ms': round(statistics.pstdev(samples), 6),
        }
