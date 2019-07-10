import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import wandb
import scipy.misc

def interface(img):
    dir="./wandb/model_3/"
    model=Sequential()
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same',
                                input_shape=(32, 32, 3)))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
    model.load_weights(dir + "model-best.h5")
    small_image = np.zeros((1, 32, 32, 3))
    small_image[0] = np.array(img) / 255.0
    print(small_image)
    pred = model.predict(small_image)
    for i, o in enumerate(pred):
        img = np.concatenate([o * 255], axis=1)
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img_name = 'test.jpg'
        img.save(img_name, 'jpeg')
