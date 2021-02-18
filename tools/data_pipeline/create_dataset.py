import os
import numpy as np
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython.display as display

SEED = 42
path = "./demo_dataset/images/"
IMG_SIZE_X = 880
IMG_SIZE_Y = 487
AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 5


def augment_fn(image):
    data = {"image":image}
    aug_data =  transforms(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img/255.0, tf.float32)
    aug_img = tf.image.resize(aug_img, size=[IMG_SIZE_Y, IMG_SIZE_X])
    return aug_img


def parse_image(img_path: str) -> dict:
    print(img_path)
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path,".jpg", ".gt.png")
    print(mask_path)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [880,487])
    mask = tf.image.convert_image_dtype(mask, tf.uint8)

    return {'image': image, 'mask': mask}
def display_sample(display_list):
    plt.figure(figsize=(18,18))

    title = ['input Image', 'Mask']

    for i in range(len(display_list)):
        plt.subplot(1,len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE_Y, IMG_SIZE_X))
    input_mask = tf.image.resize(datapoint['mask'], (IMG_SIZE_Y, IMG_SIZE_X))

    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """ Resacale pixel value betwoon 0.0 and 1.0
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


if __name__ == "__main__":
    
    train_dataset = tf.data.Dataset.list_files(path + "*.jpg", seed=SEED)
    train_dataset = train_dataset.map(parse_image)

    dataset = {'train': train_dataset}

    train = dataset['train'].map(load_image_train, num_parallel_calls=AUTOTUNE)
    train = train.cache().shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train = train.prefetch(buffer_size=tf.data.AUTOTUNE)

    for image, mask in train.take(1):
        sample_image, sample_mask = image, mask
    
    display_sample([sample_image[0], sample_mask[0]])


def display_sample(display_list):
    plt.figure(figsize=(18,18))

    title = ['Input Image', 'True Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.prepocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()



