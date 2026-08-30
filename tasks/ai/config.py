"""
Configuración de inferencia para el prototipo P0 RGB Baseline — Fold 4.

Este archivo define únicamente constantes del contrato de entrada/salida del modelo.
No carga el checkpoint ni depende de Django.
"""

MODEL_ID = "aptos2019_p0_rgb_efficientnetb0_fold4"
EXPERIMENT_NAME = "P0 RGB baseline"
CV_FOLD = 4

ARCHITECTURE = "torchvision EfficientNetB0"
NUM_CLASSES = 5
INPUT_SIZE = 224
INPUT_CHANNELS = 3

# Orden de clases usado durante entrenamiento en APTOS 2019.
CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

# Normalización ImageNet utilizada por el baseline.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Metadatos útiles para trazabilidad del prototipo.
INPUT_REPRESENTATION = "RGB"
GEOMETRY = "direct resize"
RESIZE_INTERPOLATION = "cv2.INTER_AREA"
NORMALIZATION = "ImageNet mean/std"
CHECKPOINT_SELECTION = "minimum validation loss"
MSAG_ENABLED = False
