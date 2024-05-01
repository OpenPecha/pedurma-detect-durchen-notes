from pathlib import Path

import keras
from ultralytics import YOLO

import config

class NoteClassifier:

    def __init__(self, model_path: Path):
        self.model = keras.saving.load_model(model_path)
        self.image_size = (20, 20)

    def predict(self, image_path: Path):
        img = keras.utils.load_img(str(image_path), target_size=self.image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
        predictions = self.model.predict(img_array, verbose=0)
        score = float(keras.ops.sigmoid(predictions[0][0]))
        return score

note_classifier = NoteClassifier(model_path=config.NOTE_CLASSIFIER_MODEL_PATH)
        

class NoteDetection:

    def __init__(self, model_path: Path):
        self.model = YOLO(model_path)
        self.image_size = 1472

    def predict(self, image_path: Path):
        result = self.model.predict(image_path, imgsz=self.image_size)
        orig_img_h, orig_img_w = result[0].orig_shape
        print(result[0].orig_shape)
        for box in result[0].boxes:
            x_center, y_center, w, h = box.xywh[0]
            x = int(round(float(x_center) - float(w) / 2, 0))
            y = int(round(float(y_center) - float(h) / 2, 0))
            w = int(round(float(w), 0))
            h = int(round(float(h), 0))
            yield x, y, w, h

note_detector = NoteDetection(model_path=config.NOTE_DETECTION_MODEL_PATH)