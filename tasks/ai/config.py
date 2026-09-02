"""
Configuración de inferencia para P2 Green + CLAHE — Fold 2.

Contrato de deployment derivado del checkpoint y del protocolo congelado P2.
"""

MODEL_ID = "aptos2019_p2_green_clahe_efficientnetb0_fold2"
EXPERIMENT_NAME = "P2 Green + CLAHE + EfficientNetB0 pretrained"
CANDIDATE_CODE = "P2"
CV_FOLD = 2

ARCHITECTURE = "torchvision EfficientNetB0"
NUM_CLASSES = 5
INPUT_SIZE = 224
INPUT_CHANNELS = 3
MSAG_ENABLED = False

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

# P2 FROZEN: Green -> CLAHE pre-resize -> resize -> replicate 3ch.
SOURCE_CHANNEL_INDEX = 1
SOURCE_CHANNEL_NAME = "green / RGB index 1"
CLAHE_ENABLED = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
OPERATION_ORDER = "pre_resize"
GEOMETRY = "direct_resize"
RESIZE_INTERPOLATION = "cv2.INTER_AREA"
CHANNEL_REPLICATION = "[G_CLAHE,G_CLAHE,G_CLAHE] after resize"
MEDIAN_KERNEL = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMALIZATION = "ImageNet mean/std"
CHECKPOINT_SELECTION = "minimum validation loss"

PREPROCESSING_PROTOCOL_SHA256 = (
    "ef1312c25d83a2ea8058a846f78d78334b3907d6bdbfec554fbd3748c5028623"
)
