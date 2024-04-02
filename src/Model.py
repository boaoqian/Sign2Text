import tensorflow as tf
import numpy as np
import json
import cv2
import mediapipe as mp


class Model:
    def __init__(self, model):
        self.model = tf.lite.Interpreter(f"model/Sign2Text-{model}.tflite")
        self.FRAME_LEN = 128
        with open("model/character_to_prediction_index.json", "r") as f:
            character_map = json.load(f)
        self.rev_character_map = {j: i for i, j in character_map.items()}
        self.found_signatures = list(self.model.get_signature_list().keys())
        self.prediction_fn = self.model.get_signature_runner("serving_default")

    # Function to resize and add padding.
    def resize_pad(self, x):
        x = tf.convert_to_tensor(x)
        if tf.shape(x)[0] < self.FRAME_LEN:
            x = tf.pad(x, ([[0, self.FRAME_LEN - tf.shape(x)[0]], [0, 0], [0, 0]]))
        else:
            x = tf.image.resize(x, (self.FRAME_LEN, tf.shape(x)[1]))
            x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        return x

    def predict(self, x):
        x = self.resize_pad(x)
        x = tf.reshape(x, [128, 78])
        output = self.prediction_fn(inputs=x)
        prediction_str = "".join([self.rev_character_map.get(s, "") for s in np.argmax(output['outputs'], axis=1)])
        return prediction_str
