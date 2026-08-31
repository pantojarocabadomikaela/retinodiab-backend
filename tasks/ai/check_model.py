"""Smoke test del checkpoint P2 Green + CLAHE — Fold 2."""

import torch
import torchvision

from .config import INPUT_CHANNELS, INPUT_SIZE, NUM_CLASSES
from .model_loader import get_model_bundle


def main() -> None:
    print("=== P2 Fold 2 — model loading smoke test ===")
    print("Runtime PyTorch:", torch.__version__)
    print("Runtime TorchVision:", torchvision.__version__)

    bundle = get_model_bundle()
    model = bundle.model
    device = bundle.device

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Checkpoint:", bundle.checkpoint_path)
    print("Device:", device)
    print("Best epoch:", bundle.best_epoch)
    print("Best val loss:", bundle.best_val_loss)
    print("Training PyTorch:", bundle.training_torch_version)
    print("Training TorchVision:", bundle.training_torchvision_version)
    print("Total params:", f"{total_params:,}")
    print("Trainable params:", f"{trainable_params:,}")
    print("Classifier:", model.classifier)

    if total_params != 4_013_953:
        raise RuntimeError(
            f"Parámetros incorrectos: {total_params:,}; esperado: 4,013,953."
        )

    x = torch.zeros(
        (1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        logits = model(x)

    print("Dummy input:", tuple(x.shape))
    print("Logits:", tuple(logits.shape))
    print("Logits finite:", bool(torch.isfinite(logits).all().item()))

    if tuple(logits.shape) != (1, NUM_CLASSES):
        raise RuntimeError(f"Shape de logits incorrecto: {tuple(logits.shape)}.")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Los logits contienen NaN o Inf.")

    print("\nMODEL LOADING TEST: OK")


if __name__ == "__main__":
    main()
