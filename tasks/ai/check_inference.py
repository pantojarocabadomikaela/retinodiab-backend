"""
Prueba end-to-end P2 Fold 2.

Uso:
    python -m tasks.ai.check_inference "test_images/imagen.png" --repeat 5
"""

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

from .config import CLASS_NAMES
from .inference import predict_image
from .model_loader import get_model_bundle
from .preprocessing import preprocess_image_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat debe ser >= 1.")

    path = args.image.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"No se encontró: {path}")
    image_bytes = path.read_bytes()

    print("=== P2 Fold 2 — end-to-end inference test ===")
    print("Image:", path)
    print("Image bytes:", f"{len(image_bytes):,}")

    t0 = perf_counter()
    tensor = preprocess_image_bytes(image_bytes)
    print("Preprocessed tensor:", tuple(tensor.shape))
    print("Tensor dtype:", tensor.dtype)
    print("Preprocessing time:", f"{(perf_counter()-t0)*1000:.2f} ms")

    t0 = perf_counter()
    bundle = get_model_bundle()
    print("Device:", bundle.device)
    print("Model load/get time:", f"{(perf_counter()-t0)*1000:.2f} ms")

    times=[]
    result=None
    for _ in range(args.repeat):
        t0=perf_counter()
        result=predict_image(image_bytes)
        times.append((perf_counter()-t0)*1000)

    print("\nPrediction")
    print("----------")
    print("Class ID:", result.class_id)
    print("Label:", result.label)

    print("\nProbabilities")
    print("-------------")
    for i,p in enumerate(result.probabilities):
        print(f"{i} | {CLASS_NAMES[i]:<18} {p:.6f} ({p*100:6.2f}%)")
    print("Sum:", f"{sum(result.probabilities):.12f}")

    print("\nTiming")
    print("------")
    print("Runs:", args.repeat)
    print("First:", f"{times[0]:.2f} ms")
    if len(times)>1:
        print("Mean:", f"{mean(times):.2f} ms")
        print("Mean excluding first:", f"{mean(times[1:]):.2f} ms")

    print("\nEND-TO-END INFERENCE TEST: OK")


if __name__ == "__main__":
    main()
