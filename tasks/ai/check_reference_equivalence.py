"""
Comprobación de equivalencia:
predicciones guardadas durante validation del notebook P0 Fold 4
vs.
backend local de inferencia.

Uso recomendado desde la raíz del backend:

    python -m tasks.ai.check_reference_equivalence

Por defecto espera:

    test_reference/validation_predictions_best_fold4.csv
    test_images/

También se pueden indicar rutas manualmente:

    python -m tasks.ai.check_reference_equivalence \
        --csv "ruta/al/validation_predictions_best_fold4.csv" \
        --image-dir "ruta/a/imagenes"

La comparación usa las probabilidades guardadas en:

    prob_class_0 ... prob_class_4

y exige además que el argmax/backend coincida con y_pred del CSV.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CLASS_NAMES, NUM_CLASSES
from .inference import predict_image


DEFAULT_CSV = Path("test_reference") / "validation_predictions_best_fold4.csv"
DEFAULT_IMAGE_DIR = Path("test_images")

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

REQUIRED_COLUMNS = (
    ["id_code", "y_true", "y_pred"]
    + [f"prob_class_{i}" for i in range(NUM_CLASSES)]
)


class ReferenceEquivalenceError(RuntimeError):
    """Error de configuración o integridad de la prueba de referencia."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara P0 Fold 4 local contra "
            "validation_predictions_best_fold4.csv."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=(
            "CSV de referencia. Default: "
            "test_reference/validation_predictions_best_fold4.csv"
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Carpeta con imágenes. Default: test_images/",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Tolerancia absoluta de np.allclose. Default: 1e-5.",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Tolerancia relativa de np.allclose. Default: 1e-5.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help=(
            "Máximo de imágenes coincidentes a evaluar. "
            "0 = todas. Default: 0."
        ),
    )

    return parser


def _read_reference_csv(csv_path: Path) -> List[dict]:
    csv_path = csv_path.expanduser().resolve()

    if not csv_path.is_file():
        raise ReferenceEquivalenceError(
            f"No se encontró el CSV:\n{csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ReferenceEquivalenceError(
                "El CSV no contiene encabezados."
            )

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in reader.fieldnames
        ]

        if missing:
            raise ReferenceEquivalenceError(
                "Faltan columnas requeridas en el CSV: "
                + ", ".join(missing)
            )

        rows = list(reader)

    if not rows:
        raise ReferenceEquivalenceError(
            "El CSV de referencia no contiene filas."
        )

    ids = [str(row["id_code"]).strip() for row in rows]

    if any(not item for item in ids):
        raise ReferenceEquivalenceError(
            "El CSV contiene algún id_code vacío."
        )

    if len(ids) != len(set(ids)):
        raise ReferenceEquivalenceError(
            "El CSV contiene id_code duplicados."
        )

    return rows


def _index_images(image_dir: Path) -> Dict[str, Path]:
    image_dir = image_dir.expanduser().resolve()

    if not image_dir.is_dir():
        raise ReferenceEquivalenceError(
            f"No se encontró la carpeta de imágenes:\n{image_dir}"
        )

    image_index: Dict[str, Path] = {}

    for path in image_dir.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        image_id = path.stem

        if image_id in image_index:
            raise ReferenceEquivalenceError(
                "Hay más de una imagen con el mismo stem/id_code: "
                f"{image_id}\n"
                f" - {image_index[image_id]}\n"
                f" - {path}"
            )

        image_index[image_id] = path

    return image_index


def _parse_reference_row(row: dict) -> Tuple[int, int, np.ndarray]:
    try:
        y_true = int(row["y_true"])
        y_pred = int(row["y_pred"])

        probs = np.asarray(
            [
                float(row[f"prob_class_{i}"])
                for i in range(NUM_CLASSES)
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ReferenceEquivalenceError(
            f"Fila inválida para id_code={row.get('id_code')!r}."
        ) from exc

    if probs.shape != (NUM_CLASSES,):
        raise ReferenceEquivalenceError(
            "Vector de probabilidades de referencia con shape incorrecto."
        )

    if not np.isfinite(probs).all():
        raise ReferenceEquivalenceError(
            f"Probabilidades no finitas para id_code={row['id_code']}."
        )

    reference_argmax = int(np.argmax(probs))

    if reference_argmax != y_pred:
        raise ReferenceEquivalenceError(
            f"CSV inconsistente para id_code={row['id_code']}: "
            f"argmax(probabilities)={reference_argmax}, y_pred={y_pred}."
        )

    return y_true, y_pred, probs


def _print_needed_ids(rows: List[dict], n: int = 10) -> None:
    print(
        "\nNo hay imágenes de test_images/ cuyo nombre coincida "
        "con un id_code del CSV."
    )
    print(
        "\nCopia algunas imágenes originales de APTOS a test_images/ "
        "manteniendo su nombre original. Por ejemplo:"
    )

    for row in rows[:n]:
        print(f"  {row['id_code']}.png")


def main() -> None:
    args = _parser().parse_args()

    if args.atol < 0 or args.rtol < 0:
        raise SystemExit("--atol y --rtol deben ser >= 0.")

    if args.max_images < 0:
        raise SystemExit("--max-images debe ser >= 0.")

    rows = _read_reference_csv(args.csv)
    image_index = _index_images(args.image_dir)

    rows_by_id = {
        str(row["id_code"]).strip(): row
        for row in rows
    }

    common_ids = sorted(
        set(rows_by_id).intersection(image_index)
    )

    print("=== P0 Fold 4 — notebook/backend equivalence ===")
    print("Reference CSV:", args.csv.expanduser().resolve())
    print("Reference rows:", len(rows))
    print("Image directory:", args.image_dir.expanduser().resolve())
    print("Images indexed:", len(image_index))
    print("Matching validation images:", len(common_ids))
    print("atol:", args.atol)
    print("rtol:", args.rtol)

    if not common_ids:
        _print_needed_ids(rows)
        print("\nEQUIVALENCE TEST: NOT RUN (no matching images)")
        return

    if args.max_images:
        common_ids = common_ids[: args.max_images]

    all_probability_close = True
    all_predictions_match = True
    global_max_abs_delta = 0.0

    print("\nPer-image comparison")
    print("--------------------")

    for image_id in common_ids:
        row = rows_by_id[image_id]
        image_path = image_index[image_id]

        y_true, reference_y_pred, reference_probs = (
            _parse_reference_row(row)
        )

        result = predict_image(image_path.read_bytes())

        backend_probs = np.asarray(
            result.probabilities,
            dtype=np.float64,
        )

        abs_delta = np.abs(
            backend_probs - reference_probs
        )

        max_abs_delta = float(abs_delta.max())
        global_max_abs_delta = max(
            global_max_abs_delta,
            max_abs_delta,
        )

        probs_close = bool(
            np.allclose(
                backend_probs,
                reference_probs,
                atol=args.atol,
                rtol=args.rtol,
            )
        )

        pred_match = (
            result.class_id == reference_y_pred
        )

        all_probability_close &= probs_close
        all_predictions_match &= pred_match

        status = (
            "OK"
            if probs_close and pred_match
            else "FAIL"
        )

        print(
            f"\n[{status}] {image_id}"
        )
        print(
            f"  y_true: {y_true} "
            f"({CLASS_NAMES.get(y_true, 'UNKNOWN')})"
        )
        print(
            "  Reference y_pred:",
            reference_y_pred,
            CLASS_NAMES.get(reference_y_pred, "UNKNOWN"),
        )
        print(
            "  Backend y_pred:  ",
            result.class_id,
            result.label,
        )
        print(
            "  Max |Δ probability|:",
            f"{max_abs_delta:.12e}",
        )

        for class_id in range(NUM_CLASSES):
            print(
                f"    class {class_id}: "
                f"ref={reference_probs[class_id]:.10f}  "
                f"backend={backend_probs[class_id]:.10f}  "
                f"Δ={abs_delta[class_id]:.3e}"
            )

    print("\nSummary")
    print("-------")
    print("Images evaluated:", len(common_ids))
    print(
        "Prediction matches:",
        "OK" if all_predictions_match else "FAIL",
    )
    print(
        "Probability allclose:",
        "OK" if all_probability_close else "FAIL",
    )
    print(
        "Global max |Δ probability|:",
        f"{global_max_abs_delta:.12e}",
    )

    if all_predictions_match and all_probability_close:
        print(
            "\nNOTEBOOK <-> BACKEND EQUIVALENCE: OK"
        )
    else:
        print(
            "\nNOTEBOOK <-> BACKEND EQUIVALENCE: FAIL"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
