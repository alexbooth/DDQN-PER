import keras
import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.optimizers import RMSprop

def rgb2gray(pil_image):
    """ converts PIL image to grayscale np array """
    return np.array(pil_image.convert('L'))

def load_model(json, weights, optimizer, loss, learning_rate, gradient_clip):
    json_file = open(json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights)
    rmsprop = RMSprop(lr=learning_rate, clipvalue=gradient_clip)
    model.compile(loss=loss, optimizer=rmsprop)
    return model

def save_model(model, json_filename, weight_filename):
    model_json = model.to_json()
    with open(json_filename, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weight_filename)   

def save_image(np_array):
    im = Image.fromarray(np_array, 'L')
    im.save("im.png")
