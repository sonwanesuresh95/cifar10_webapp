from tensorflow import keras
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image


class CIFAR10:
    def __init__(self):
        self.model = keras.models.load_model(r'model/cifar10_model')
        self.mapping = {0: 'airplane',
                        1: 'automobile',
                        2: 'bird',
                        3: 'cat',
                        4: 'deer',
                        5: 'dog',
                        6: 'frog',
                        7: 'horse',
                        8: 'ship',
                        9: 'truck'}

    def predict_category(self, image):
        image = image[:, :, :3]
        image = image.reshape(1, 32, 32, 3)
        pred = self.model.predict(image)
        category = self.mapping[np.where(pred[0] == max(pred[0]))[0][0]]
        try:
            return category
        except Exception as e:
            return str(e)
