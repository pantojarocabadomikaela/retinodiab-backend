from io import BytesIO

import numpy as np
import torch
from django.test import SimpleTestCase
from PIL import Image

from .ai.config import INPUT_CHANNELS, INPUT_SIZE
from .ai.preprocessing import (
    ImagePreprocessingError,
    decode_rgb_image,
    deterministic_preprocess_p2,
    preprocess_image_bytes,
)


def create_png_bytes(width=16, height=12):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = 20
    rgb[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :] * 10
    rgb[:, :, 2] = 200

    buffer = BytesIO()
    Image.fromarray(rgb, mode='RGB').save(buffer, format='PNG')
    return buffer.getvalue()


class ImageDecodingTests(SimpleTestCase):
    def test_decode_rgb_image_rejects_empty_file(self):
        with self.assertRaises(ImagePreprocessingError):
            decode_rgb_image(b'')

    def test_decode_rgb_image_rejects_non_image_bytes(self):
        with self.assertRaises(ImagePreprocessingError):
            decode_rgb_image(b'esto no es una imagen')


class PreprocessingTests(SimpleTestCase):
    def test_deterministic_preprocess_produces_expected_shape_and_channels(self):
        rgb = decode_rgb_image(create_png_bytes())

        result = deterministic_preprocess_p2(rgb)

        self.assertEqual(
            result.shape,
            (INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS),
        )
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    def test_deterministic_preprocess_rejects_invalid_input(self):
        invalid_rgb = np.zeros((10, 10, 3), dtype=np.float32)

        with self.assertRaises(ImagePreprocessingError):
            deterministic_preprocess_p2(invalid_rgb)

    def test_preprocess_image_bytes_returns_normalized_finite_tensor(self):
        tensor = preprocess_image_bytes(create_png_bytes())

        self.assertEqual(
            tuple(tensor.shape),
            (1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE),
        )
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all().item())
