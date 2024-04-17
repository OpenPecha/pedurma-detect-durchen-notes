from pathlib import Path

import keras

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
        

