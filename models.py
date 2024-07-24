import os
from pathlib import Path
from typing import List

import keras
import numpy as np
from ultralytics import YOLO

import config

os.environ["KERAS_BACKEND"] = "tensorflow"


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


class DoubleTsekClassifier:

    def __init__(self, model_path: Path):
        self.model = keras.saving.load_model(model_path)
        self.image_size = (23, 15)

    def predict(self, image_path: Path):
        img = keras.utils.load_img(str(image_path), target_size=self.image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
        predictions = self.model.predict(img_array, verbose=0)
        score = float(keras.ops.sigmoid(predictions[0][0]))
        return score


double_tsek_classifier = DoubleTsekClassifier(
    model_path=config.DOUBLE_TSEK_CLASSIFIER_MODEL_PATH
)


class YoloDetector:

    def __init__(self, model_path: Path, image_size=1472):
        self.model = YOLO(model_path)
        self.image_size = image_size

    def mod_box(self, box, x_adjust, y_adjust):
        x_center, y_center, w, h = box.xywh[0]
        x = int(round(float(x_center) - float(w) / 2, 0)) + x_adjust
        y = int(round(float(y_center) - float(h) / 2, 0)) + y_adjust
        w = int(round(float(w), 0))
        h = int(round(float(h), 0))
        return x, y, w, h

    def predict(self, image_path: Path, x_adjust=0, y_adjust=0):
        result = self.model.predict(image_path, imgsz=self.image_size, verbose=False)
        for box in result[0].boxes:
            x, y, w, h = self.mod_box(box, x_adjust, y_adjust)
            yield x, y, w, h

    def batch_predict(self, images_path: List[Path], x_adjust=0, y_adjust=0):
        results = self.model.predict(images_path, imgsz=self.image_size, verbose=False)
        for result in results:
            boxes = []
            for box in result.boxes:
                boxes.append(self.mod_box(box, x_adjust, y_adjust))
            yield boxes


note_detector = YoloDetector(model_path=config.NOTE_DETECTION_MODEL_PATH)
double_tsek_detector = YoloDetector(model_path=config.DOUBLE_TSEK_DETECTION_MODEL_PATH)


class NoteNumRecoginiser:

    def __init__(self, model_path: Path):
        self.model = keras.saving.load_model(model_path)
        self.image_size = (20, 20)
        self.class_names = [
            "1",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "2",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "3",
            "30",
            "31",
            "32",
            "33",
            "34",
            "35",
            "36",
            "37",
            "38",
            "39",
            "4",
            "40",
            "41",
            "42",
            "43",
            "44",
            "45",
            "46",
            "47",
            "48",
            "49",
            "5",
            "50",
            "51",
            "52",
            "53",
            "54",
            "55",
            "56",
            "57",
            "58",
            "59",
            "6",
            "60",
            "61",
            "62",
            "64",
            "7",
            "8",
            "9",
        ]

    def predit(self, image_path):
        img = keras.utils.load_img(str(image_path), target_size=self.image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
        predictions = self.model.predict(img_array, verbose=0)
        return self.class_names[np.argmax(predictions[0])]


note_num_recogniser = NoteNumRecoginiser(
    model_path=config.NOTE_NUM_RECOGNIZER_MODEL_PATH
)
