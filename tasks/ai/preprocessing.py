"""
Preprocessing determinista de inferencia para P2 Green + CLAHE — Fold 2.

Replica el pipeline de VALIDATION del notebook P2:

PIL.Image.open(...).convert("RGB")
-> green = rgb[:, :, 1]
-> CLAHE(clipLimit=2.0, tileGridSize=(8, 8)) PRE-RESIZE
-> cv2.resize(..., 224x224, INTER_AREA)
-> [G_CLAHE, G_CLAHE, G_CLAHE]
-> HWC -> CHW
-> float32 / 255.0
-> ImageNet mean/std
-> batch dimension

No aplica augmentations ni filtro mediano.
"""

from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_CHANNELS,
    INPUT_SIZE,
    SOURCE_CHANNEL_INDEX,
)


class ImagePreprocessingError(ValueError):
    """Error controlado al decodificar o transformar una imagen."""


def decode_rgb_image(image_bytes: bytes) -> np.ndarray:
    """Decodifica bytes con Pillow exactamente como el dataset del notebook."""
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

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ImagePreprocessingError(
            f"Se esperaba RGB HxWx3; se recibió shape={rgb.shape}."
        )

    return np.ascontiguousarray(rgb, dtype=np.uint8)


def deterministic_preprocess_p2(rgb: np.ndarray) -> np.ndarray:
    """
    Replica literalmente deterministic_preprocess_p2() del notebook.

    Returns
    -------
    np.ndarray
        uint8 HWC con shape [224, 224, 3]. Los tres canales son idénticos.
    """
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
    ):
        raise ImagePreprocessingError("Se espera RGB uint8 HxWx3.")

    green = rgb[:, :, SOURCE_CHANNEL_INDEX]

    clahe = cv2.createCLAHE(
        clipLimit=float(CLAHE_CLIP_LIMIT),
        tileGridSize=tuple(CLAHE_TILE_GRID),
    )

    # FROZEN: CLAHE antes del resize.
    green_clahe = clahe.apply(green)

    green_clahe_224 = cv2.resize(
        green_clahe,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    green_clahe_3ch = np.repeat(
        green_clahe_224[..., None],
        INPUT_CHANNELS,
        axis=2,
    )

    output = np.ascontiguousarray(green_clahe_3ch, dtype=np.uint8)

    expected_shape = (INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS)
    if output.shape != expected_shape:
        raise RuntimeError(
            f"Shape P2 inesperado: {output.shape}; esperado: {expected_shape}."
        )

    # Contrato P2: los tres canales deben ser copias exactas.
    if not (
        np.array_equal(output[:, :, 0], output[:, :, 1])
        and np.array_equal(output[:, :, 1], output[:, :, 2])
    ):
        raise RuntimeError("P2 produjo canales replicados no idénticos.")

    return output


def p2_uint8_to_normalized_tensor(p2_image: np.ndarray) -> torch.Tensor:
    """Convierte P2 uint8 HWC a tensor CHW normalizado con ImageNet."""
    expected_shape = (INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS)
    if (
        not isinstance(p2_image, np.ndarray)
        or p2_image.dtype != np.uint8
        or p2_image.shape != expected_shape
    ):
        raise ImagePreprocessingError(
            f"Se esperaba P2 uint8 shape={expected_shape}; "
            f"se recibió dtype={getattr(p2_image, 'dtype', None)}, "
            f"shape={getattr(p2_image, 'shape', None)}."
        )

    image = torch.from_numpy(
        np.ascontiguousarray(p2_image.transpose(2, 0, 1))
    ).to(torch.float32) / 255.0

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    return (image - mean) / std


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Pipeline público: bytes -> tensor float32 [1, 3, 224, 224]."""
    rgb = decode_rgb_image(image_bytes)
    p2_image = deterministic_preprocess_p2(rgb)
    tensor = p2_uint8_to_normalized_tensor(p2_image).unsqueeze(0)

    expected_shape = (1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"Shape de preprocessing inesperado: {tuple(tensor.shape)}; "
            f"esperado: {expected_shape}."
        )
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"dtype inesperado: {tensor.dtype}.")
    if not torch.isfinite(tensor).all():
        raise RuntimeError("El tensor preprocesado contiene NaN o Inf.")

    return tensor
