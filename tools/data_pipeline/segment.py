import os
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
    
    CLASSES = ['path']
    
    def __init__(
            self, 
            x_ps: List[Path], 
            y_ps: List[Path], 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.x_ps = x_ps
        self.y_ps = y_ps

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(str(self.x_ps[i].resolve()))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (448,640))
        mask = cv2.imread(str(self.y_ps[i].resolve()), 0)
        mask = cv2.resize(mask, (448,640))
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary

        
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
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

    def get_preprocessing(preprocessing_fn):

    
        _transform = [
            A.Lambda(image=preprocessing_fn),
        ]
        return A.Compose(_transform)

if __name__ == "__main__":
    # sm.set_framework('keras')
    # print(tf.keras.utils.Sequence)
    # print(keras.utils.Sequence)
    # print(tf.keras.utils.Sequence == keras.utils.Sequence )
    DATA_DIR = Path('demo_dataset')
    x = list(DATA_DIR.glob('*.jpg'))
    y = list(DATA_DIR.glob('*.gt.png'))

    assert(len(x) == len(y))

    x.sort()
    y.sort()
    valid_index = math.floor(len(x) * 0.7)
    test_index = math.floor(len(x) * 0.85)
    x_train = x[:valid_index]
    x_valid = x[valid_index:test_index]
    x_test = x[test_index:]
    print(len(x_train))
    print(len(x_valid))
    print(len(x_test))
    
    y_train = y[:valid_index]
    y_valid = y[valid_index:test_index]
    y_test = y[test_index:]

    dataset = Dataset(x_train, y_train, classes=['path'])

    # image, mask = dataset[5] # get some sample
    # visualize(
    #     image=image, 
    #     path_mask=mask[..., 0].squeeze(),
    # )

    BACKBONE = 'efficientnetb3'
    BATCH_SIZE = 8
    CLASSES = ['car']
    LR = 0.0001
    EPOCHS = 40
    
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    train_dataset = Dataset(
        x_train, 
        y_train, 
        classes=['path']
    )

    valid_dataset = Dataset(
            x_valid, 
            y_valid, 
        classes=['path']
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(train_dataloader[0][0].shape)
    assert train_dataloader[0][0].shape == (BATCH_SIZE, 640, 448, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, 640, 448, n_classes)

    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    
    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader), 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        validation_data=valid_dataloader, 
        validation_steps=len(valid_dataloader),
    )
    