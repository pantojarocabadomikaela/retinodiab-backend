"""
Prueba manual end-to-end de inferencia para P0 Fold 4.

Uso desde la raíz del backend:

    python -m tasks.ai.check_inference "ruta/a/imagen.png"

Opcionalmente:

    python -m tasks.ai.check_inference "ruta/a/imagen.png" --repeat 5

`--repeat` permite medir varias inferencias consecutivas. La primera puede
ser más lenta por inicialización/carga del modelo.
"""

import argparse
from pathlib import Path
from time import perf_counter
from statistics import mean

from .config import CLASS_NAMES
from .inference import predict_image
from .model_loader import get_model_bundle
from .preprocessing import preprocess_image_bytes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test end-to-end de P0 Fold 4."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Ruta a una retinografía JPG/PNG.",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Número de inferencias a ejecutar para medir latencia (default: 1).",
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat debe ser >= 1.")

    image_path = args.image.expanduser().resolve()

    if not image_path.is_file():
        raise SystemExit(f"No se encontró la imagen: {image_path}")

    image_bytes = image_path.read_bytes()

    print("=== P0 Fold 4 — end-to-end inference test ===")
    print("Image:", image_path)
    print("Image bytes:", f"{len(image_bytes):,}")

    # 1. Validar preprocessing por separado.
    t0 = perf_counter()
    tensor = preprocess_image_bytes(image_bytes)
    preprocessing_ms = (perf_counter() - t0) * 1000.0

    print("Preprocessed tensor:", tuple(tensor.shape))
    print("Tensor dtype:", tensor.dtype)
    print("Preprocessing time:", f"{preprocessing_ms:.2f} ms")

    # 2. Cargar modelo antes de medir las inferencias.
    t0 = perf_counter()
    bundle = get_model_bundle()
    model_load_ms = (perf_counter() - t0) * 1000.0

    print("Device:", bundle.device)
    print("Model load/get time:", f"{model_load_ms:.2f} ms")

    # 3. Inferencia completa. predict_image vuelve a ejecutar preprocessing
    # deliberadamente porque queremos probar exactamente la función que
    # después llamará Django.
    times_ms = []
    last_result = None

    for _ in range(args.repeat):
        start = perf_counter()
        last_result = predict_image(image_bytes)
        elapsed_ms = (perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    result = last_result

    print("\nPrediction")
    print("----------")
    print("Class ID:", result.class_id)
    print("Label:", result.label)

    print("\nProbabilities")
    print("-------------")

    for class_id, probability in enumerate(result.probabilities):
        label = CLASS_NAMES[class_id]
        print(
            f"{class_id} | {label:<18} "
            f"{probability:.6f} ({probability * 100:6.2f}%)"
        )

    probability_sum = sum(result.probabilities)
    print("Sum:", f"{probability_sum:.12f}")

    print("\nLogits")
    print("------")
    for class_id, logit in enumerate(result.logits):
        print(f"{class_id}: {logit:.8f}")

    print("\nTiming")
    print("------")
    print("Runs:", args.repeat)
    print("Each run includes preprocessing + model inference.")
    print("First:", f"{times_ms[0]:.2f} ms")

    if args.repeat > 1:
        print("Mean:", f"{mean(times_ms):.2f} ms")
        print("Min:", f"{min(times_ms):.2f} ms")
        print("Max:", f"{max(times_ms):.2f} ms")

        if len(times_ms) > 1:
            print(
                "Mean excluding first:",
                f"{mean(times_ms[1:]):.2f} ms",
            )

    print("\nEND-TO-END INFERENCE TEST: OK")


if __name__ == "__main__":
    main()
