# Model

## Prior art
Before we started developing this project we looked online for research and/or papers that have done something similar to what we were planning on doing. We found two researches, one by Giusti et al[^1]. and one by Smolyanskiy et al.[^2] Giusti classified their images with 1 of 3 classes: left view, right view and center view. They trained their model to classify an image with one of these classes to then get steering from this prediction.

Smolyanskiy did the same things but instead of 3 classes, they used 6 classes. The 3 same classes as Giusti and 3 classes that represented if the drone was in the center, left or right of the path. The addition of these 3 classes was done to make sure that the drone always flies in the middle of the path and so to avoid obstacles to the side of the road.

You can keep adding classes to make a better steering estimation but this is not the best way to do this. We came with the idea to use image segmentation to create an exact mask of where the road is on the camera's field of view. This way it is possible to make more accurate steering estimations and it even gives us the possibility to implement a system that is able to take turns.

## Semantic segmentation
Semantic segmentation is like object detection a computer vision technique. With object detection you are trying to detect a certain object in an image and most of the time draw a box around it. Semantic segmentation kind of does the same but it tries to label a region in an image. In our case, we are only interested where the road is, so we only need to mark the road and no other regions.

```{figure} https://i.imgur.com/VKcyK1y.png
:name: ref

[Semantic segmentation](https://medium.com/deelvin-machine-learning/compression-artifact-localization-as-a-semantic-segmentation-task-2b57f8a4a022)
```




## Datasets

To train our network we used three datasets: Freiburg, Giusti and Steentjes. After processing these datasets into our wanted format, every dataset consisted of RGB images and their corresponding binary mask image.


### Freiburg (DeepScene)
Freiburg (DeepScene)[^3] is a dataset created by the university of Freiburg and contains mostly images of gravel roads. All these images have a manually labeled segmentation mask for multiple classes like: sky,road,high vegetation etc. Since we were only interested in the trail, we made a script that converted the original mask into a binary mask where the road was white and all the rest was black. This dataset has +-250 labeled images. The original dataset also contained depth maps and other ground truth masks. These images were captured using a rover

![](https://i.imgur.com/eq4iVmJ.jpg)



### Giusti
The Giusti dataset is a dataset of forest trails in Switzerland. This dataset did not contain segmentation masks but it was classified as 1 of 3 classes. They captured these images by placing 3 action camera's on a helmet and then walking over forest trails in Switzerland. Since we wanted to extract a mask from the path, we had to segment these images ourselves. The original dataset consisted out of 20 000 images, a lot of them were almost the exact same image since they were frames extracted from a video. We created a script that made a selection of the images based on their color histogram. Images that followed each other and had almost the exact same histogram were deleted. To do this we set up a web tool that gave a simple interface to segment the images. Labeling 500+ images is a very time consuming task so we contacted some teachers if it was possible to join one of their classes and let the first and second year students help us with labeling images. The tool however returns a json file with the coordinates of all points of the mask. We created a script that converted the json file into an image file of the same size as the original image with again, the path in white and all the rest in black. In total we have +-580 labeled images.

![](https://i.imgur.com/DKwvPBt.jpg)


### Steentjes
Our last dataset, Steentjes, was a dataset with images from a forest in Schiplaken that we took ourselves. We created the masks for these images using the same tool as mentioned before. This dataset has 77 labeled images.

We also experimented with combining multiple datasets together. After some testing we got the best result from combining Freiburg and Steentjes together.

All these datasets were converted into TFRecords to work on the Google Cloud TPUs and were stored on Google Cloud Buckets to maximize training performance.

![](https://i.imgur.com/YN2LeuL.jpg)


## Augmentations
To train a machine learning model you need lots and lots of ground truth data. Sometimes it is easy to collect the data and more data often means better results from your model. But in our case collecting and annotating the data is very labour intensive. Luckily there is a technique called data augmentations that takes a piece of data, in our case an image and the mask, and does some augmentations to it to create new data. Some examples of augmentations you can perform on images are: horizontal flip, crop, hue, color, etc.

One important thing when doing augmentations is that you have to use some type of randomization to augment the image. When you apply the same augmentation to every image, the results are not going to change much since the model overfits to that type of data. When you use a randomizer that sets some parameters in the augmentations you have a more varied set of augmentations and thus a more varied dataset.

We first wanted to use a library called ‘Albumentations’. This library has an easy to use API to perform augmentations on images and it had way better performance then the built-in Tensorflow augmentations. Unfortunately we were not able to use this library because it is not possible to run arbitrary Python code on a TPU. The TPUs only accept Tensorflow commands.
![](https://i.imgur.com/lSsEWCk.png)


## Architecture
We chose to implement our model in [Tensorflow](https://www.tensorflow.org/), a machine learning library created by Google. The reason we chose Tensorflow is because it gave us the possibility to use a Keras model via the Tensorflow Keras API. This enabled us to use Google TPUs to train our model. Tensorflow also has a very active community and extensive documentation. 

The network structure that we chose is called **Unet**[^4].  This network is created by the university of Freiburg and it was originally created to segment clinical images like X-rays. Unet is a so called 'Convolutional neural network'. The reason why we chose for this model is because it gave us the perfect balance between performance and accuracy. It is also a model that does not require a very large dataset to get good results from.

If we talk about a network structure, we don't mean a network with routers and switches but a network of operations that happen on a image and that are connected to each other. We are not going in full detail about every layer in this network but there are two main ones: Convolution layer and Max Pooling.



We used a high level API called **Segmentation_models**[^5] to create our model. This library abstracts away some things and makes it easier to implement a model. This library returns a Keras model but since we were planning on using Google’s TPU’s to train our model, we had to use the Tensorflow Keras API.

As the backbone for our network we used **efficientnetb3**[^6]. This is the encoder used to extract features from the data. It might be a bit confusing but we are not actually using the standard Unet but we are implementing efficientnet in the Unet structure. If you look at the image below, you can see that there are two parts to the Unet: Downsample and upsample. The downsample part is the efficientnet and the upsamle part is efficientnet but reversed. You can substitute efficientnet for other backbones like Mobilenet which is optimized to run on mobile devices.
![]()
```{figure} https://i.imgur.com/1A02S1r.png
:name: ref

[Unet](https://github.com/qubvel/segmentation_models)
```

The loss function we used was a custom loss function that was also implemented in the Segmentation models library. Essentially it is two loss functions, dice loss and Focal Binary Loss, added together.

**Dice loss** is a binary semantic segmentation specific loss function. 

**Focal binary loss**, penalizes hard to classify images more heavily relative to easy to classify images.

## Transfer Learning

Since our dataset was not that big we used a pre trained model and we did transfer learning on this model to get better results from it. With transfer learning you load a pre-trained model into the network and then start training the model on your data. The pre-trained model is already capable of extracting abstract features from an image so we only have to train for specific features in our data. The pre-trained model is trained on the 2012 ILSVRC ImageNet dataset.

We also tried performing transfer learning by training our model on one of our three datasets to then apply transfer learning on one of the other datasets. Unfortunately we didn’t get any useful results by doing this.

## Training

All datasets were split into three subsets, train, test and validation. The splits we used were: 70% training, 15% test and 15% validation. An image can never be in 2 splits at the same time.

We did all our training without augmentations in this [Jupyter Notebook](train.ipynb)

All training with augmentations was done [here](train_aug.ipynb)

Most of the experiments were done with a batch size of 16 images. We tried a batch size of 32 once but it did not give us better results.

When you train your model to the point that it is getting too specific and it does not generalize anymore, your validation loss will start to rise again. This is because the model is so ‘used’ to the training data that when you use other data, it does not perform that well anymore. To prevent this from happening and to decrease training times, we implemented **early stopping**. Essentially what early stopping does is that when your validation loss starts to rise again, it will stop training and save the best model. We used a ‘patience’ of 3, this means that when your model does not improve for 3 consecutive epochs, it will stop the training. You can let the early stopping monitor all metrics, we chose to monitor validation loss since it is the most accurate metric.

![](https://i.imgur.com/LYZu3qg.png)


### Freiburg
When training our model on the Freiburg dataset, we got some pretty good results but the downside was that it only performed well on images that looked like the images in the Freiburg dataset. When we predicted images from the other datasets on this model we didn’t get any good results at all, it was only good to classify paths with a distinct color change like in the Freiburg dataset.

When training on Freiburg + augmentations we didn’t see a big improvement since the dataset on itself was large and varied enough.

### Giusti
Training on Giusti was an exciting one for us since we didn’t know if the masks we got from our crowdsourced project were sufficient to train a model. After training and tweaking some parameters we did not have a very good result. This is probably due to the fact that the Giusti dataset is too varied and that there was too much noisy data. When testing the model on it’s own validation set it didn’t give good results at all, most of the masks were off or there were no masks at all. Also the testing on Freiburg and Steentjes didn’t give good results at all.

Adding augmentations to the training didn’t help either, it gave pretty much the same result as the model without augmentations.
![](https://i.imgur.com/m0VhhPA.png)



### Steentjes
The Steentjes dataset without augmentations was a total miss. Instead of the validation loss going down, the validation loss went up drastically. The explanation for this is simple, the dataset is way too small. When you look at the masks that the model predicted from the test dataset, you can see that the model sees the path but the problem is that the model is not developed enough to rule out the rest of the image that is no path. If we had more data in this dataset, the model could have been a good one.

We also added some augmentations to this dataset and here you can see a big difference. Thanks to the extra augmentations the validation loss dropped. For our drone we don’t really need all the detail in the mask, if it can see the direction the path is going in, it should be fine. But if you want to do other things with the mask, you need a better defined path, not just the rough direction. 
![](https://i.imgur.com/Ntcr8Qy.png)


### Giusti + Freiburg + Steentjes
We also tried to concatenate some datasets together to see if this gives any good or interesting results. The concatenation of Giusti, Freiburg and Steentjes was ok in seeing the path but it sometimes lacked detail in more difficult images.

Since this dataset was pretty big it didn’t really get any benefits from doing augmentations.

With this dataset we also tried a different batch size then we were using before. The batch size in most of the tests was 16 but for this one we also tried a batch size of 32. We didn’t see any improvements in the validation loss so we stuck with the batch size of 16 for the following tests.
![](https://i.imgur.com/wQePWd6.png)


### Freiburg + Steentjes

Finally we tried a concatenation of Freiburg and Steentjes. The first time we ran this training, it kept improving until all the 40 epochs were finished and the training stopped. We upped the number of epochs so it could train for more than 40 epochs and in the second run it still improved and it ended at 0.24 validation loss. When looking at the output from the test data We saw some pretty good results, paths were well defined, also horizontal paths that don’t start at the bottom of the image.

Like with all tests, we also tested if augmentations would help and in this case it made a good model better. After training this model with the augmentations we saw very good results. Most predictions were well detailed and even very difficult trails were segmented correctly. Paths to the side of the road were also detected by the model, this can be especially useful for further development of the software to let the drone take turns. Also the model almost always classifies non roads as non roads so that our drone does not start moving in the wrong direction.
![](https://i.imgur.com/Se2r6mR.png)


## TPU 
Training a deep learning models requires a lot of processing power. You can train your model on a CPU but this might take ages. Nowadays people use GPU’s (Graphical processing unit) to train deep learning models. The advantage of using a GPU over a CPU is that a GPU can process large amounts of data (higher bandwidth). Training a deep learning model essentially is doing lots of matrix multiplications and these matrices can be really big. GPU’s like the name says are actually created for processing graphics and are not really optimized for training deep learning models. That’s where [Google’s TPUs](https://cloud.google.com/tpu) come in to play. These TPUs are basically GPU’s that are optimized for training deep learning models. You can’t just go out and buy a TPU but you can only hire one on Google’s Cloud Platform. We were able to get one month of TPU usage for free. 

Without this TPU we wouldn’t have been able to do so much experimenting with training our model on different datasets etc. because it would have taken day’s to train our model on our own laptop.


```{figure} https://i.imgur.com/f5sTiPD.png
:name: ref

[Google cloud TPU](https://cloud.google.com/tpu)
```



## TensorRT
[TensorRT](https://developer.nvidia.com/tensorrt) is an SDK made by Nvidia for doing inference on a deep learning model that is optimized for Nvidia GPUs. By converting your model from a Tensorflow Keras model to a TensorRT model, you first have to freeze the model. Freezing a model is essentially locking all the weights in the model so that your model stays the same. Then you can convert your model to TensorRT.

The reason why we wanted to convert our model to TensorRT is that we were planning on running the inference on a Nvidia Jetson Nano 4GB. This way we could get a performance boost for the inference. The Jetson nano is a mini computer that has a 128-core Maxwell GPU. Unfortunately we were not able to make the conversion work.

```{figure} https://i.imgur.com/eD0p8QL.png
:name: ref

[TensorRT](https://developer.nvidia.com/tensorrt)
```



## Real life testing

When testing our final Freiburg+Steentjes+augmentations model in a real forest we had very good results. Our drone always stayed on the path and followed it nicely. The model also performed well on densely overgrown paths and was able to guide our drone over this path. When pointing the drone in a direction where there is no path, the model returns a black image, indicating that it is well trained to not classify non-paths as paths. The model was not only capable of detecting the path that starts at the bottom of the screen but also when the path was horizontal. We tested this by going a few meters off trail and pointing it in the direction of the path. The drone was able to see the path even with some trees in the way.

To load a model you can use this [Jupyter notebook](load_model.ipynb) and do predictions yourselves.

<iframe width="560" height="315" src="https://www.youtube.com/embed/eeTxom6YrDs" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<br>

[^1]: Smolyanskiy et al., “Toward Low-Flying Autonomous MAV Trail Navigation using DeepNeural Networks for Environmental Awareness” May 2017.  

[^2]: Giusti et al., “A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots” Dec 2015.

[^3]: A. Valada et al. “Deep Multispectral Semantic Scene Understanding of Forested Environments Using Multimodal Fusion,” 2016.

[^4]: Ronneberger et al., “U-Net: Convolutional Networks for Biomedical Image Segmentation.” May 2015.

[^5]: P. Yakubovskiy, “qubvel/segmentation_models,” 2019. [Online]. Available: https://github.com/qubvel/segmentation_models. 

[^6]: M. Tan, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks” May 2019.
