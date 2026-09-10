from rest_framework import status
from rest_framework.test import APITestCase
from django.contrib.auth.hashers import check_password

from .models import User


class UserTests(APITestCase):
	def test_post_hashes_password_and_does_not_return_it(self):
		user_data = {
			'email': 'nuevo@example.com',
			'nombre': 'Nuevo Usuario',
			'password': 'una-clave-segura',
			'rol': 'paciente',
			'diabetes': False,
			'fecha_nacimiento': '2000-01-01T00:00:00Z',
		}

		response = self.client.post('/api/v1/users/', user_data, format='json')

		self.assertEqual(response.status_code, status.HTTP_201_CREATED)
		user = User.objects.get(email=user_data['email'])
		self.assertNotEqual(user.password, user_data['password'])
		self.assertTrue(check_password(user_data['password'], user.password))
		self.assertNotIn('password', response.data)

	def test_authenticate_checks_hashed_password(self):
		user = User.objects.create(
			email='login@example.com',
			nombre='Login User',
			password='clave-correcta',
			rol='paciente',
			diabetes=False,
			fecha_nacimiento='2000-01-01T00:00:00Z',
		)

		self.assertEqual(User.authenticate(user.email, 'clave-correcta'), user)
		self.assertIsNone(User.authenticate(user.email, 'clave-incorrecta'))
		self.assertIsNone(User.authenticate('inexistente@example.com', 'clave'))

	def test_post_rejects_duplicate_email(self):
		user_data = {
			'email': 'existing@example.com',
			'nombre': 'Existing User',
			'password': 'password',
			'rol': 'user',
			'diabetes': False,
			'fecha_nacimiento': '2000-01-01T00:00:00Z',
		}
		User.objects.create(**user_data)

		response = self.client.post('/api/v1/users/', user_data, format='json')

		self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
		self.assertEqual(response.data['email'][0], 'email duplicado')

	def test_patch_allows_same_email_for_same_user(self):
		user = User.objects.create(
			email='andres2@gmail.com',
			nombre='Andres Arispe Medina',
			password='password',
			rol='paciente',
			diabetes=True,
			fecha_nacimiento='2024-04-29T00:45:07.543976Z',
		)

		payload = {
			'nombre': 'Andres Arispe Medina',
			'email': 'andres2@gmail.com',
			'rol': 'paciente',
			'diabetes': True,
			'fecha_nacimiento': '2024-04-29T00:45:07.543976Z',
		}

		response = self.client.patch(f'/api/v1/users/{user.id}/', payload, format='json')

		self.assertEqual(response.status_code, status.HTTP_200_OK)
		self.assertEqual(response.data['email'], 'andres2@gmail.com')
