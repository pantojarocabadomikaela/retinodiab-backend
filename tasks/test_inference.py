from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from django.test import SimpleTestCase

from .ai.config import MODEL_ID
from .ai.inference import InferenceError, predict_image


class InferenceTests(SimpleTestCase):
    def setUp(self):
        self.input_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    def test_predict_image_returns_highest_probability_class(self):
        model = Mock(
            return_value=torch.tensor(
                [[0.0, 1.0, 5.0, 2.0, -1.0]],
                dtype=torch.float32,
            )
        )
        bundle = SimpleNamespace(model=model, device=torch.device('cpu'))

        with (
            patch(
                'tasks.ai.inference.preprocess_image_bytes',
                return_value=self.input_tensor,
            ),
            patch('tasks.ai.inference.get_model_bundle', return_value=bundle),
        ):
            result = predict_image(b'imagen-simulada')

        self.assertEqual(result.model_id, MODEL_ID)
        self.assertEqual(result.class_id, 2)
        self.assertEqual(result.label, 'Moderate')
        self.assertAlmostEqual(sum(result.probabilities), 1.0, places=6)
        model.assert_called_once()

    def test_predict_image_wraps_model_failure(self):
        model = Mock(side_effect=RuntimeError('fallo interno'))
        bundle = SimpleNamespace(model=model, device=torch.device('cpu'))

        with (
            patch(
                'tasks.ai.inference.preprocess_image_bytes',
                return_value=self.input_tensor,
            ),
            patch('tasks.ai.inference.get_model_bundle', return_value=bundle),
            self.assertRaisesRegex(InferenceError, 'forward pass'),
        ):
            predict_image(b'imagen-simulada')

    def test_predict_image_rejects_unexpected_logits_shape(self):
        model = Mock(return_value=torch.zeros((2, 5), dtype=torch.float32))
        bundle = SimpleNamespace(model=model, device=torch.device('cpu'))

        with (
            patch(
                'tasks.ai.inference.preprocess_image_bytes',
                return_value=self.input_tensor,
            ),
            patch('tasks.ai.inference.get_model_bundle', return_value=bundle),
            self.assertRaisesRegex(InferenceError, 'Shape de logits inesperado'),
        ):
            predict_image(b'imagen-simulada')
