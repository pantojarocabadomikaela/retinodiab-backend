"""
Preprocessing determinista para inferencia con P0 RGB Baseline — Fold 4.

Replica el pipeline de VALIDATION del notebook de entrenamiento:

PIL.Image.open(...)
→ convert("RGB")
→ np.asarray(...)
→ cv2.resize(..., (224, 224), interpolation=cv2.INTER_AREA)
→ HWC -> CHW
→ float32 / 255.0
→ normalización ImageNet
→ dimensión batch

No aplica augmentations, crop de FOV, circle crop, CLAHE, canal verde
ni filtrado mediano.
"""

from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_CHANNELS,
    INPUT_SIZE,
)


class ImagePreprocessingError(ValueError):
    """Error controlado al decodificar o transformar una imagen de entrada."""


def decode_rgb_image(image_bytes: bytes) -> np.ndarray:
    """
    Decodifica bytes de imagen como RGB uint8.

    Se usa Pillow para reproducir el método de carga empleado por el
    dataset del notebook baseline.
    """
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ImagePreprocessingError(
            "La imagen debe recibirse como bytes o bytearray."
        )

    if len(image_bytes) == 0:
        raise ImagePreprocessingError("El archivo de imagen está vacío.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImagePreprocessingError(
            "No se pudo decodificar el archivo como una imagen válida."
        ) from exc

    # np.asarray(PIL.Image) puede devolver memoria de solo lectura.
    # La copia garantiza un ndarray contiguo y seguro para OpenCV/PyTorch.
    return np.ascontiguousarray(rgb)


def resize_rgb_224(rgb: np.ndarray) -> np.ndarray:
    """
    Aplica el mismo direct resize del baseline:
    cv2.resize(..., 224x224, INTER_AREA).
    """
    if not isinstance(rgb, np.ndarray):
        raise ImagePreprocessingError("La imagen RGB debe ser un numpy.ndarray.")

    if rgb.ndim != 3 or rgb.shape[2] != INPUT_CHANNELS:
        raise ImagePreprocessingError(
            f"Se esperaba una imagen RGB HxWx{INPUT_CHANNELS}; "
            f"se recibió shape={getattr(rgb, 'shape', None)}."
        )

    resized = cv2.resize(
        rgb,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    return np.ascontiguousarray(resized)


def rgb_to_normalized_tensor(rgb: np.ndarray) -> torch.Tensor:
    """
    Convierte RGB uint8 HWC en tensor float32 CHW y normaliza con ImageNet.
    """
    if rgb.shape != (INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS):
        raise ImagePreprocessingError(
            "La imagen debe estar redimensionada antes de convertirla a tensor. "
            f"Shape recibido: {rgb.shape}."
        )

    # Replica:
    # torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    image = torch.from_numpy(
        np.ascontiguousarray(rgb.transpose(2, 0, 1))
    )

    image = image.to(torch.float32) / 255.0

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=torch.float32,
    )[:, None, None]

    std = torch.tensor(
        IMAGENET_STD,
        dtype=torch.float32,
    )[:, None, None]

    image = (image - mean) / std

    return image


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """
    Pipeline público para inferencia.

    Parameters
    ----------
    image_bytes:
        Contenido binario del JPG/PNG u otra imagen compatible con Pillow.

    Returns
    -------
    torch.Tensor
        Tensor float32 con shape [1, 3, 224, 224], listo para el modelo.
    """
    rgb = decode_rgb_image(image_bytes)
    rgb = resize_rgb_224(rgb)
    tensor = rgb_to_normalized_tensor(rgb)

    # EfficientNetB0 espera NCHW.
    tensor = tensor.unsqueeze(0)

    expected_shape = (1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"Shape de preprocessing inesperado: {tuple(tensor.shape)}; "
            f"esperado: {expected_shape}."
        )

    return tensor
