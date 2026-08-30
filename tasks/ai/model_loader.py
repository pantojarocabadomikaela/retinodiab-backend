"""
Carga reproducible de P0 RGB Baseline — Fold 4.

Responsabilidades:
- reconstruir torchvision EfficientNetB0 sin descargar pesos;
- sustituir classifier[1] por Linear(1280, 5);
- cargar model_state_dict desde best_val_loss.pt con strict=True;
- validar metadatos esenciales del checkpoint;
- seleccionar CPU/CUDA;
- mantener el modelo cargado en memoria mediante lazy loading.

No realiza preprocessing ni inferencia de probabilidades.
"""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional, Union
import warnings

import torch
from torch import nn
import torchvision
from torchvision.models import efficientnet_b0

from .config import (
    ARCHITECTURE,
    CV_FOLD,
    EXPERIMENT_NAME,
    MSAG_ENABLED,
    NUM_CLASSES,
)


# Ruta portable respecto al propio paquete:
# tasks/ai/model_loader.py -> tasks/iamodels/p0_fold4/best_val_loss.pt
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "iamodels"
    / "p0_fold4"
    / "best_val_loss.pt"
)


class ModelLoadingError(RuntimeError):
    """Error controlado al reconstruir o cargar el modelo."""


@dataclass
class ModelBundle:
    """Modelo ya cargado junto con información útil de deployment."""

    model: nn.Module
    device: torch.device
    checkpoint_path: Path
    best_epoch: int
    best_val_loss: float
    training_torch_version: Optional[str]
    training_torchvision_version: Optional[str]


_bundle: Optional[ModelBundle] = None
_bundle_lock = Lock()


def select_device() -> torch.device:
    """
    Selecciona CUDA si está disponible; de lo contrario usa CPU.

    Para una demo académica EfficientNetB0 puede ejecutarse en CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model() -> nn.Module:
    """
    Reconstruye EXACTAMENTE la topología del baseline P0.

    Importante:
    weights=None evita una descarga de ImageNet durante deployment.
    El checkpoint ya contiene todos los pesos entrenados.
    """
    model = efficientnet_b0(weights=None)

    if not isinstance(model.classifier, nn.Sequential):
        raise ModelLoadingError(
            "La estructura de classifier de EfficientNetB0 no es la esperada."
        )

    if len(model.classifier) < 2 or not isinstance(model.classifier[1], nn.Linear):
        raise ModelLoadingError(
            "No se encontró classifier[1] como capa Linear."
        )

    in_features = model.classifier[1].in_features

    if in_features != 1280:
        raise ModelLoadingError(
            f"EfficientNetB0 inesperado: classifier in_features={in_features}, "
            "se esperaba 1280."
        )

    # Conserva classifier[0] = Dropout(p=0.2) de TorchVision.
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model


def _validate_checkpoint_metadata(checkpoint: dict) -> None:
    """
    Comprueba que el archivo parece corresponder al P0 Fold 4 esperado.

    No se usa la config del checkpoint para cambiar dinámicamente la
    arquitectura: el deployment permanece explícito y reproducible.
    """
    if "model_state_dict" not in checkpoint:
        raise ModelLoadingError(
            "El checkpoint no contiene la clave 'model_state_dict'."
        )

    config = checkpoint.get("config")

    if not isinstance(config, dict):
        raise ModelLoadingError(
            "El checkpoint no contiene un bloque 'config' válido."
        )

    observed_fold = config.get("cv_fold")
    if observed_fold != CV_FOLD:
        raise ModelLoadingError(
            f"Checkpoint de fold incorrecto: {observed_fold}; "
            f"se esperaba Fold {CV_FOLD}."
        )

    observed_experiment = config.get("experiment")
    if observed_experiment != "RGB baseline EfficientNetB0 pretrained":
        raise ModelLoadingError(
            "El checkpoint no corresponde al experimento RGB baseline esperado. "
            f"Observado: {observed_experiment!r}."
        )

    model_cfg = config.get("model", {})
    observed_architecture = model_cfg.get("architecture")
    observed_num_classes = model_cfg.get("num_classes")
    observed_msag = model_cfg.get("msag")

    if observed_architecture != ARCHITECTURE:
        raise ModelLoadingError(
            f"Arquitectura inesperada: {observed_architecture!r}; "
            f"esperada: {ARCHITECTURE!r}."
        )

    if observed_num_classes != NUM_CLASSES:
        raise ModelLoadingError(
            f"Número de clases inesperado: {observed_num_classes}; "
            f"esperado: {NUM_CLASSES}."
        )

    if bool(observed_msag) != MSAG_ENABLED:
        raise ModelLoadingError(
            f"Estado MSAG inesperado: {observed_msag}; "
            f"esperado: {MSAG_ENABLED}."
        )


def _warn_if_runtime_differs(checkpoint: dict) -> None:
    """
    Informa si PyTorch/TorchVision difieren de las versiones de entrenamiento.

    Una diferencia de versión no implica por sí sola incompatibilidad.
    La prueba decisiva es que load_state_dict(strict=True) y el forward pass
    funcionen correctamente.
    """
    config = checkpoint.get("config", {})

    train_torch = config.get("torch_version")
    train_torchvision = config.get("torchvision_version")

    runtime_torch = torch.__version__
    runtime_torchvision = torchvision.__version__

    if train_torch and train_torch != runtime_torch:
        warnings.warn(
            "Versión de PyTorch distinta a la de entrenamiento: "
            f"training={train_torch}, runtime={runtime_torch}.",
            RuntimeWarning,
            stacklevel=2,
        )

    if train_torchvision and train_torchvision != runtime_torchvision:
        warnings.warn(
            "Versión de TorchVision distinta a la de entrenamiento: "
            f"training={train_torchvision}, runtime={runtime_torchvision}.",
            RuntimeWarning,
            stacklevel=2,
        )


def load_model_bundle(
    checkpoint_path: Union[Path, str] = DEFAULT_CHECKPOINT_PATH,
    device: Optional[torch.device] = None,
) -> ModelBundle:
    """
    Reconstruye y carga una instancia nueva del modelo.

    Esta función es útil para tests. Para el servidor Django debe preferirse
    get_model_bundle(), que reutiliza una única instancia en memoria.
    """
    checkpoint_path = Path(checkpoint_path).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "No se encontró el checkpoint en:\n"
            f"{checkpoint_path}\n\n"
            "Ubicación esperada por defecto:\n"
            "tasks/iamodels/p0_fold4/best_val_loss.pt"
        )

    if device is None:
        device = select_device()

    try:
        # El archivo es un checkpoint propio y confiable del proyecto.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise ModelLoadingError(
            f"No se pudo leer el checkpoint: {checkpoint_path}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ModelLoadingError(
            "El contenido del .pt no es el diccionario de checkpoint esperado."
        )

    _validate_checkpoint_metadata(checkpoint)
    _warn_if_runtime_differs(checkpoint)

    model = build_model()

    try:
        # strict=True es deliberado: no aceptamos capas faltantes o sobrantes.
        incompatible = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
    except Exception as exc:
        raise ModelLoadingError(
            "El model_state_dict no es compatible con la arquitectura "
            "EfficientNetB0 P0 esperada."
        ) from exc

    # Con strict=True normalmente este objeto ya estará vacío, pero se
    # comprueba de forma explícita para documentar el contrato.
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ModelLoadingError(
            "Carga no estricta detectada inesperadamente. "
            f"missing_keys={incompatible.missing_keys}; "
            f"unexpected_keys={incompatible.unexpected_keys}"
        )

    model.to(device)
    model.eval()

    config = checkpoint.get("config", {})

    return ModelBundle(
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        best_epoch=int(checkpoint["epoch"]),
        best_val_loss=float(checkpoint["best_val_loss"]),
        training_torch_version=config.get("torch_version"),
        training_torchvision_version=config.get("torchvision_version"),
    )


def get_model_bundle() -> ModelBundle:
    """
    Lazy loading thread-safe para Django.

    Primera llamada:
        construye + carga el modelo.

    Llamadas posteriores:
        reutilizan la misma instancia residente en RAM/VRAM.
    """
    global _bundle

    if _bundle is None:
        with _bundle_lock:
            if _bundle is None:
                _bundle = load_model_bundle()

    return _bundle
