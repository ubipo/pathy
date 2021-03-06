import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow.python.keras.backend import dtype
from visualise import show_two, show
from dataset import load_dataset, split_dataset_paths

model = tf.keras.models.load_model('./test_save.h5', compile=False)

IMAGE_SIZE = [448, 640]
x = []
img = cv2.imread('./giusti.jpg')
img.resize(IMAGE_SIZE)
x.append(np.asarray(img,dtype="int32"))
x=np.array(x)
print(x.shape)


prediction = model(np.asarray(x))

show({
    "RGB image": img,
    "Prediction": prediction,
})