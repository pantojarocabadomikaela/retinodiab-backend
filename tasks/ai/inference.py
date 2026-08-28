"""
Inferencia para P0 RGB Baseline — Fold 4.

Este módulo une:
    image bytes
        -> preprocessing determinista
        -> EfficientNetB0
        -> logits
        -> softmax
        -> clase predicha + probabilidades

No conoce nada de Django, HTTP, base de datos ni serializers.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch

from .config import CLASS_NAMES, MODEL_ID, NUM_CLASSES
from .model_loader import get_model_bundle
from .preprocessing import preprocess_image_bytes


class InferenceError(RuntimeError):
    """Error controlado durante la inferencia."""


@dataclass(frozen=True)
class InferenceResult:
    """
    Resultado estructurado de una predicción.

    `probabilities` conserva el orden de clases 0..4 utilizado
    durante entrenamiento.
    """

    model_id: str
    class_id: int
    label: str
    probabilities: List[float]
    logits: List[float]
    device: str

    def probabilities_by_label(self) -> Dict[str, float]:
        return {
            CLASS_NAMES[i]: float(self.probabilities[i])
            for i in range(NUM_CLASSES)
        }

    def to_dict(self, include_logits: bool = False) -> dict:
        """
        Representación lista para JsonResponse/serializer.

        Por defecto no expone logits, porque no son necesarios
        para el frontend.
        """
        result = {
            "model_id": self.model_id,
            "prediction": {
                "class_id": self.class_id,
                "label": self.label,
            },
            "probabilities": self.probabilities_by_label(),
            "device": self.device,
        }

        if include_logits:
            result["logits"] = list(self.logits)

        return result


def predict_image(image_bytes: bytes) -> InferenceResult:
    """
    Ejecuta una predicción completa sobre una única imagen.

    Parameters
    ----------
    image_bytes:
        Contenido binario de la imagen original subida por el usuario.

    Returns
    -------
    InferenceResult
        Clase predicha, probabilidades softmax y logits.

    Notes
    -----
    - No se aplican augmentations.
    - No se aplica circle crop, CLAHE, canal verde ni filtro mediano.
    - La entrada se procesa con el pipeline P0 congelado.
    """
    tensor = preprocess_image_bytes(image_bytes)

    bundle = get_model_bundle()
    model = bundle.model
    device = bundle.device

    tensor = tensor.to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )

    try:
        with torch.inference_mode():
            logits_tensor = model(tensor)
            probabilities_tensor = torch.softmax(logits_tensor, dim=1)
    except Exception as exc:
        raise InferenceError("Falló el forward pass del modelo.") from exc

    expected_shape = (1, NUM_CLASSES)

    if tuple(logits_tensor.shape) != expected_shape:
        raise InferenceError(
            f"Shape de logits inesperado: {tuple(logits_tensor.shape)}; "
            f"esperado: {expected_shape}."
        )

    if tuple(probabilities_tensor.shape) != expected_shape:
        raise InferenceError(
            "Shape de probabilidades inesperado: "
            f"{tuple(probabilities_tensor.shape)}."
        )

    if not torch.isfinite(logits_tensor).all():
        raise InferenceError("Los logits contienen NaN o Inf.")

    if not torch.isfinite(probabilities_tensor).all():
        raise InferenceError("Las probabilidades contienen NaN o Inf.")

    probabilities_cpu = (
        probabilities_tensor[0]
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )

    logits_cpu = (
        logits_tensor[0]
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )

    probability_sum = float(probabilities_cpu.sum().item())

    if abs(probability_sum - 1.0) > 1e-6:
        raise InferenceError(
            f"Softmax inválido: suma de probabilidades={probability_sum:.12f}."
        )

    class_id = int(torch.argmax(probabilities_cpu).item())

    if class_id not in CLASS_NAMES:
        raise InferenceError(
            f"Índice de clase fuera del mapping configurado: {class_id}."
        )

    return InferenceResult(
        model_id=MODEL_ID,
        class_id=class_id,
        label=CLASS_NAMES[class_id],
        probabilities=[
            float(value) for value in probabilities_cpu.tolist()
        ],
        logits=[
            float(value) for value in logits_cpu.tolist()
        ],
        device=str(device),
    )
