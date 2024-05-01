from pathlib import Path

IMAGES_PATH = Path(__file__).parent.parent / "images"

# Templates
NOTE_MAKER_TEMPLATES_PATH = Path(__file__).parent / "templates" / "note_maker"
NOTE_MAKER_TEMPLATES_PATH.mkdir(parents=True, exist_ok=True)
DOUBLE_TSEK_TEMPLATES_PATH = Path(__file__).parent / "templates" / "double_tsek"
DOUBLE_TSEK_TEMPLATES_PATH.mkdir(parents=True, exist_ok=True)

# Training data paths
TRAINING_DATA_PATH = Path(__file__).parent / "traindata"
NOTE_CLASSIFIER_TRAINING_DATA = TRAINING_DATA_PATH / "note_classifier"
NOTE_CLASSIFIER_TRAINING_DATA.mkdir(parents=True, exist_ok=True)
NOTE_NUM_RECOGNITION_DATASET = TRAINING_DATA_PATH / "note_num_recogination"
NOTE_NUM_RECOGNITION_DATASET.mkdir(parents=True, exist_ok=True)
NOTE_DETECTION_TRAINING_DATA = TRAINING_DATA_PATH / "note_detection"
NOTE_DETECTION_TRAINING_DATA.mkdir(parents=True, exist_ok=True)

# Models
NOTE_CLASSIFIER_MODEL_PATH = Path(__file__).parent / "models" / "note_classifier" / "note_classifier.keras"
NOTE_CLASSIFIER_MODEL_PATH = Path(__file__).parent / "models" / "note_num_recognition" / "note_num_recognition.keras"
# NOTE_DETECTION_MODEL_PATH = Path(__file__).parent / "models" / "note_detection" / "yolov8s_train1_e10.pt"
NOTE_DETECTION_MODEL_PATH = Path(__file__).parent / "models" / "note_detection" / "yolov8s_train_e2.pt"
