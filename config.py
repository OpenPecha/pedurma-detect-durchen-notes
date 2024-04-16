from pathlib import Path

IMAGES_PATH = Path(__file__).parent.parent / "images"
TEMPLATES_PATH = Path(__file__).parent / "templates"
TEMPLATES_PATH.mkdir(parents=True, exist_ok=True)

# Training data paths
TRAINING_DATA_PATH = Path(__file__).parent / "traindata"
NOTE_CLASSIFIER_TRAINING_DATA = TRAINING_DATA_PATH / "note_classifier"
NOTE_CLASSIFIER_TRAINING_DATA.mkdir(parents=True, exist_ok=True)
NOTE_DETECTION_TRAINING_DATA = TRAINING_DATA_PATH / "note_detection"
NOTE_DETECTION_TRAINING_DATA.mkdir(parents=True, exist_ok=True)
