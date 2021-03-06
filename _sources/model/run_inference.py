import tensorflow as tf
import segmentation_models as sm
import numpy as np
import cv2

IMAGE_SIZE = [448, 640]

img = cv2.imread("steentjes.jpg")

rgb = tf.image.resize(img, IMAGE_SIZE)
rgb = tf.cast(rgb, tf.float32) / 255.0

model = tf.keras.models.load_model('./aug_steentjes_freiburg.h5', compile=False)
# print(model.summary())
predictions = model(np.asarray([rgb]))


def model_output_to_opencv_image(prediction: tf.Tensor):
    '''
    Keras model output floats in [0,1], while OpenCV represents images as uint8
    (i.e. integers in [0, 255]).
    '''
    return np.round(prediction.numpy() * 255).astype('uint8')

cv2.imwrite("prediction.png", model_output_to_opencv_image(predictions[0]))
