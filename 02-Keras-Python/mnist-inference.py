import numpy as np
import keras
from keras import layers
import tensorflow as tf
import PIL

model_file = "mnist.keras"

model = keras.models.load_model(model_file)
img = keras.utils.load_img("upside-down-dos.jpg")
img = tf.image.resize(img, (28, 28))
img = tf.image.rgb_to_grayscale(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
print(predictions)