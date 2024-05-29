from pathlib import Path

IMAGES_PATH = Path(__file__).parent.parent / "images"
INPUT_PATH = Path(__file__).parent / "input"
OUTPUT_PATH = Path(__file__).parent / "output"
INPUT_PATH.mkdir(parents=True, exist_ok=True)
IMAGES_OUTPUT_PATH = OUTPUT_PATH / "images"
OCR_OUTPUT_PATH = OUTPUT_PATH / "ocr"
IMAGES_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
OCR_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Templates
NOTE_MAKER_TEMPLATES_PATH = Path(__file__).parent / "templates" / "note_maker"
NOTE_MAKER_TEMPLATES_PATH.mkdir(parents=True, exist_ok=True)
DOUBLE_TSEK_TEMPLATES_PATH = Path(__file__).parent / "templates" / "double_tsek"
DOUBLE_TSEK_TEMPLATES_PATH.mkdir(parents=True, exist_ok=True)

# Training data paths
TRAINING_DATA_PATH = Path(__file__).parent / "traindata"

# Note
NOTE_CLASSIFIER_TRAINING_DATA = TRAINING_DATA_PATH / "note_classifier"
NOTE_CLASSIFIER_TRAINING_DATA.mkdir(parents=True, exist_ok=True)
NOTE_NUM_RECOGNITION_DATASET = TRAINING_DATA_PATH / "note_num_recogination"
NOTE_NUM_RECOGNITION_DATASET.mkdir(parents=True, exist_ok=True)
NOTE_DETECTION_TRAINING_DATA = TRAINING_DATA_PATH / "note_detection"
NOTE_DETECTION_TRAINING_DATA.mkdir(parents=True, exist_ok=True)

# double tsek
DOUBLE_TSEK_CLASSIFIER_TRAINING_DATA = TRAINING_DATA_PATH / "double_tsek_classifier"
DOUBLE_TSEK_CLASSIFIER_TRAINING_DATA.mkdir(parents=True, exist_ok=True)
DOUBLE_TSEK_DETECTION_TRAINING_DATA = TRAINING_DATA_PATH / "double_tsek_detection"
DOUBLE_TSEK_DETECTION_TRAINING_DATA.mkdir(parents=True, exist_ok=True)

# Models
MODELS_PATH = Path(__file__).parent / "models"

## Note Models
NOTE_CLASSIFIER_MODEL_PATH = MODELS_PATH / "note_classifier" / "note_classifier.keras"
NOTE_NUM_RECOGNIZER_MODEL_PATH = MODELS_PATH / "note_num_recognition" / "note_num_recognition.keras"
NOTE_DETECTION_MODEL_PATH = MODELS_PATH / "note_detection" / "yolov8s_e20.pt"

## Double-Tsek models
DOUBLE_TSEK_CLASSIFIER_MODEL_PATH = MODELS_PATH / "double_tsek_classifier" / "double_tsek_classfier.keras"
DOUBLE_TSEK_DETECTION_MODEL_PATH = MODELS_PATH / "double_tsek_detection" / "best.pt"