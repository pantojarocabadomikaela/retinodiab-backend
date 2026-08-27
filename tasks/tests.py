from rest_framework import status
from rest_framework.test import APITestCase

from .models import User


class UserTests(APITestCase):
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
		self.assertEqual(response.data['email'][0], 'duplicated email')
