import base64
from io import BytesIO
from unittest.mock import patch

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
from rest_framework import status
from rest_framework.test import APITestCase

from .ai.inference import InferenceError, InferenceResult
from .models import Diagnostico


def create_png_bytes():
    rgb = np.full((12, 16, 3), 128, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(rgb, mode='RGB').save(buffer, format='PNG')
    return buffer.getvalue()


class ImageValidationEndpointTests(APITestCase):
    def test_validate_image_rejects_request_without_file(self):
        response = self.client.post(
            '/api/v1/validate-image/',
            {'email': 'paciente@example.com', 'observaciones': ''},
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.json()['error'], 'No se recibió ninguna imagen')
        self.assertEqual(Diagnostico.objects.count(), 0)

    def test_validate_image_rejects_invalid_image(self):
        invalid_file = SimpleUploadedFile(
            'retina.png',
            b'contenido que no es una imagen',
            content_type='image/png',
        )

        response = self.client.post(
            '/api/v1/validate-image/',
            {
                'email': 'paciente@example.com',
                'observaciones': '',
                'imageFile': invalid_file,
            },
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('imagen válida', response.json()['error'])
        self.assertEqual(Diagnostico.objects.count(), 0)


class DiagnosisStorageTests(APITestCase):
    @patch('tasks.views.predict_image')
    def test_successful_inference_stores_diagnosis_and_original_image(
        self,
        mocked_predict_image,
    ):
        image_bytes = create_png_bytes()
        mocked_predict_image.return_value = InferenceResult(
            model_id='modelo-prueba',
            class_id=2,
            label='Moderate',
            probabilities=[0.01, 0.04, 0.85, 0.07, 0.03],
            logits=[0.0, 1.0, 4.0, 1.5, 0.5],
            device='cpu',
        )
        image_file = SimpleUploadedFile(
            'retina.png',
            image_bytes,
            content_type='image/png',
        )

        response = self.client.post(
            '/api/v1/validate-image/',
            {
                'email': 'paciente@example.com',
                'observaciones': 'Prueba unitaria',
                'imageFile': image_file,
            },
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(Diagnostico.objects.count(), 1)

        diagnosis = Diagnostico.objects.get()
        self.assertEqual(diagnosis.email, 'paciente@example.com')
        self.assertEqual(diagnosis.resultado, 'Moderate')
        self.assertEqual(diagnosis.observaciones, 'Prueba unitaria')
        self.assertEqual(
            base64.b64decode(diagnosis.imagen.encode('utf-8')),
            image_bytes,
        )
        self.assertEqual(response.json()['prediction']['class_id'], 2)
        mocked_predict_image.assert_called_once_with(image_bytes)

    @patch('tasks.views.predict_image')
    def test_failed_inference_does_not_store_diagnosis(
        self,
        mocked_predict_image,
    ):
        mocked_predict_image.side_effect = InferenceError('fallo simulado')
        image_file = SimpleUploadedFile(
            'retina.png',
            create_png_bytes(),
            content_type='image/png',
        )

        response = self.client.post(
            '/api/v1/validate-image/',
            {
                'email': 'paciente@example.com',
                'observaciones': '',
                'imageFile': image_file,
            },
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        self.assertEqual(Diagnostico.objects.count(), 0)
