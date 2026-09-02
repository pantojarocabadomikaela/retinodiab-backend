"""
Carga reproducible de P2 Green + CLAHE — Fold 2.

La arquitectura sigue siendo torchvision EfficientNetB0 sin MSAG.
El cambio P1 -> P2 está exclusivamente en preprocessing (CLAHE).
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
    CANDIDATE_CODE,
    CHANNEL_REPLICATION,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    CV_FOLD,
    EXPERIMENT_NAME,
    GEOMETRY,
    MEDIAN_KERNEL,
    MSAG_ENABLED,
    NORMALIZATION,
    NUM_CLASSES,
    OPERATION_ORDER,
    PREPROCESSING_PROTOCOL_SHA256,
    RESIZE_INTERPOLATION,
)


DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "iamodels"
    / "p2_fold2"
    / "best_val_loss.pt"
)


class ModelLoadingError(RuntimeError):
    """Error controlado al reconstruir o cargar el modelo."""


@dataclass
class ModelBundle:
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model() -> nn.Module:
    """Reconstruye EfficientNetB0 + Linear(1280, 5), sin descargar ImageNet."""
    model = efficientnet_b0(weights=None)

    if not isinstance(model.classifier, nn.Sequential):
        raise ModelLoadingError("Classifier de EfficientNetB0 inesperado.")
    if len(model.classifier) < 2 or not isinstance(model.classifier[1], nn.Linear):
        raise ModelLoadingError("No se encontró classifier[1] como Linear.")

    in_features = model.classifier[1].in_features
    if in_features != 1280:
        raise ModelLoadingError(
            f"classifier in_features={in_features}; se esperaba 1280."
        )

    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def _validate_checkpoint_metadata(checkpoint: dict) -> None:
    """Rechaza checkpoints que no correspondan exactamente a P2 Fold 2."""
    if "model_state_dict" not in checkpoint:
        raise ModelLoadingError("Falta 'model_state_dict' en el checkpoint.")

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ModelLoadingError("El checkpoint no contiene 'config' válido.")

    if config.get("cv_fold") != CV_FOLD:
        raise ModelLoadingError(
            f"Checkpoint de fold incorrecto: {config.get('cv_fold')}; "
            f"se esperaba Fold {CV_FOLD}."
        )
    if config.get("candidate_code") != CANDIDATE_CODE:
        raise ModelLoadingError(
            f"Candidate incorrecto: {config.get('candidate_code')!r}; "
            f"se esperaba {CANDIDATE_CODE!r}."
        )
    if config.get("experiment") != EXPERIMENT_NAME:
        raise ModelLoadingError(
            f"Experimento inesperado: {config.get('experiment')!r}."
        )

    model_cfg = config.get("model", {})
    if model_cfg.get("architecture") != ARCHITECTURE:
        raise ModelLoadingError(
            f"Arquitectura inesperada: {model_cfg.get('architecture')!r}."
        )
    if model_cfg.get("num_classes") != NUM_CLASSES:
        raise ModelLoadingError(
            f"Número de clases inesperado: {model_cfg.get('num_classes')}."
        )
    if bool(model_cfg.get("msag")) != MSAG_ENABLED:
        raise ModelLoadingError(
            f"Estado MSAG inesperado: {model_cfg.get('msag')}."
        )
    if model_cfg.get("total_parameters") not in (None, 4_013_953):
        raise ModelLoadingError(
            f"Número de parámetros registrado inesperado: "
            f"{model_cfg.get('total_parameters')}."
        )

    input_cfg = config.get("input", {})
    checks = {
        "candidate_code": (input_cfg.get("candidate_code"), CANDIDATE_CODE),
        "use_clahe": (input_cfg.get("use_clahe"), True),
        "clahe_clip_limit": (
            float(input_cfg.get("clahe_clip_limit", -1)),
            float(CLAHE_CLIP_LIMIT),
        ),
        "clahe_tile_grid": (
            tuple(input_cfg.get("clahe_tile_grid", [])),
            tuple(CLAHE_TILE_GRID),
        ),
        "median_kernel": (input_cfg.get("median_kernel"), MEDIAN_KERNEL),
        "operation_order": (input_cfg.get("operation_order"), OPERATION_ORDER),
        "geometry": (input_cfg.get("geometry"), GEOMETRY),
        "resize_interpolation": (
            input_cfg.get("resize_interpolation"),
            RESIZE_INTERPOLATION,
        ),
        "channel_replication": (
            input_cfg.get("channel_replication"),
            CHANNEL_REPLICATION,
        ),
        "normalization": (input_cfg.get("normalization"), NORMALIZATION),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise ModelLoadingError(
                f"Metadata P2 incompatible en {name}: "
                f"observed={observed!r}, expected={expected!r}."
            )

    lock = config.get("preprocessing_protocol_lock", {})
    observed_hash = lock.get("canonical_sha256")
    if observed_hash != PREPROCESSING_PROTOCOL_SHA256:
        raise ModelLoadingError(
            "SHA256 del protocolo de preprocessing no coincide: "
            f"{observed_hash!r}."
        )


def _warn_if_runtime_differs(checkpoint: dict) -> None:
    config = checkpoint.get("config", {})
    train_torch = config.get("torch_version")
    train_torchvision = config.get("torchvision_version")

    if train_torch and train_torch != torch.__version__:
        warnings.warn(
            "Versión de PyTorch distinta a la de entrenamiento: "
            f"training={train_torch}, runtime={torch.__version__}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if train_torchvision and train_torchvision != torchvision.__version__:
        warnings.warn(
            "Versión de TorchVision distinta a la de entrenamiento: "
            f"training={train_torchvision}, runtime={torchvision.__version__}.",
            RuntimeWarning,
            stacklevel=2,
        )


def load_model_bundle(
    checkpoint_path: Union[Path, str] = DEFAULT_CHECKPOINT_PATH,
    device: Optional[torch.device] = None,
) -> ModelBundle:
    checkpoint_path = Path(checkpoint_path).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "No se encontró el checkpoint en:\n"
            f"{checkpoint_path}\n\n"
            "Ubicación esperada por defecto:\n"
            "tasks/iamodels/p2_fold2/best_val_loss.pt"
        )

    if device is None:
        device = select_device()

    try:
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
        raise ModelLoadingError("El .pt no contiene el diccionario esperado.")

    _validate_checkpoint_metadata(checkpoint)
    _warn_if_runtime_differs(checkpoint)

    model = build_model()
    try:
        incompatible = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
    except Exception as exc:
        raise ModelLoadingError(
            "model_state_dict incompatible con EfficientNetB0 P2."
        ) from exc

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ModelLoadingError(
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
    global _bundle
    if _bundle is None:
        with _bundle_lock:
            if _bundle is None:
                _bundle = load_model_bundle()
    return _bundle
