from rest_framework import status
from rest_framework.test import APITestCase

from .models import User


class AuthenticationEndpointTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create(
            email='login@example.com',
            nombre='Usuario Login',
            password='clave-correcta',
            rol='paciente',
            diabetes=False,
            fecha_nacimiento='2000-01-01T00:00:00Z',
        )

    def test_validate_credentials_accepts_valid_password(self):
        response = self.client.post(
            '/api/v1/validate-credentials/',
            {'email': self.user.email, 'password': 'clave-correcta'},
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()['id'], self.user.id)
        self.assertNotIn('password', response.json())

    def test_validate_credentials_rejects_invalid_password(self):
        response = self.client.post(
            '/api/v1/validate-credentials/',
            {'email': self.user.email, 'password': 'clave-incorrecta'},
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.json()['mensaje'], 'Credenciales inválidas')

    def test_validate_credentials_rejects_unknown_email(self):
        response = self.client.post(
            '/api/v1/validate-credentials/',
            {'email': 'inexistente@example.com', 'password': 'cualquier-clave'},
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.json()['mensaje'], 'Credenciales inválidas')
