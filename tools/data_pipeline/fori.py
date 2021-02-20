import os, sys
from re import X
import cv2
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from typing import List
import segmentation_models as sm
import albumentations as A
import tensorflow as tf

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title('hello')
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(448, 640)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)
    
    
# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """

    def __init__(
            self, 
            x_ps: List[Path], 
            y_ps: List[Path],  
            augmentation=None, 
            preprocessing=None,
    ):
        self.x_ps = x_ps
        self.y_ps = y_ps
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(str(self.x_ps[i].resolve()))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = (640,448)
        image = cv2.resize(image, size)
        mask = cv2.imread(str(self.y_ps[i].resolve()), 0)
        mask = cv2.resize(mask, size)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask != 0)]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.x_ps)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        return self.dataset[i]
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return tuple(batch)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


def init_tpu():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://10.240.1.2')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.TPUStrategy(resolver)


DATA_DIR = Path("/opt/padnet/freiburg-smol")
x = list(DATA_DIR.glob('*.jpg'))
y = list(DATA_DIR.glob('*.gt.png'))

assert(len(x) == len(y))
assert(len(x) > 0)

x.sort()
y.sort()
valid_index = math.floor(len(x) * 0.7)
test_index = math.floor(len(x) * 0.85)
x_train = x[:valid_index]
x_valid = x[valid_index:test_index]
x_test = x[test_index:]

y_train = y[:valid_index]
y_valid = y[valid_index:test_index]
y_test = y[test_index:]

BACKBONE = 'efficientnetb3'
BATCH_SIZE = 1
CLASSES = ['path']
LR = 0.0001
EPOCHS = 40

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

tpu_strategy = init_tpu()

# with open("/dev/random"):
with tpu_strategy.scope():    
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    optim = keras.optimizers.Adam(LR)

    # metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    print("All devices: ", tf.config.list_logical_devices('TPU'))
    model: keras.Model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    # model.compile(optim, total_loss, metrics)
    model.compile(optim, total_loss)

preprocess_input = sm.get_preprocessing(BACKBONE)

train_dataset = Dataset(
    x_train, 
    y_train, 
    # augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

valid_dataset = Dataset(
        x_test, 
        y_test, 
    # augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

train_data = list(train_dataloader)
valid_data = list(valid_dataloader)

x_train, y_train = zip(*train_data)
x_valid, y_valid = zip(*valid_data)
# dimensions are: (batch_nr, img_x, img_y, rgb)

# model.predict(x_train[0])

# x_train_one_batch = x_train[0]
# y_train_one_batch = y_train[0]

# model.train_on_batch(x_train_one_batch, y_train_one_batch)

history = model.fit(
    x=np.asarray(x_train),
    y=np.asarray(y_train),
    batch_size=1,
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=(np.asarray(x_valid), np.asarray(y_valid)), 
)
