# Datasets

To train our network we used three datasets: Freiburg, Giusti and Steentjes. After processing these datasets into our own format, every dataset consisted of RGB images and their corresponding binary mask image.


## Freiburg

Freiburg (DeepScene)[^3] is a dataset created by the university of Freiburg and contains mostly images of gravel roads. All these images have a manually labeled segmentation mask for multiple classes like: sky, road, high vegetation etc. Since we were only interested in the trail, we made [a script to convert the original mask](scripts/convert_masks) to a binary mask where the road was white and all the rest was black. This dataset has +-250 labeled images. The original dataset also contained depth maps and other ground truth masks. These images were captured using a rover.

We created [Freiburg flatten](scripts/freiburg_flatten) to flatten the dataset into our
own format.

```{figure} media/freiburg-example.jpg
Freiburg dataset example
```


## Giusti

The Giusti dataset is a dataset of forest trails in Switzerland. This dataset did not contain segmentation masks but it was classified as 1 of 3 classes. They captured these images by placing 3 action camera's on a helmet and then walking over forest trails in Switzerland. Since we wanted to extract a mask from the path, we had to segment these images ourselves. The original dataset consisted out of 20 000 images, a lot of them were almost the exact same image since they were frames extracted from a video. We created a script that made a selection of the images based on their color histogram. Images that followed each other and had almost the exact same histogram were deleted. To do this we set up a web tool that gave a simple interface to segment the images. Labeling 500+ images is a very time consuming task so we contacted some teachers if it was possible to join one of their classes and let the first and second year students help us with labeling images. The tool however returns a json file with the coordinates of all points of the mask. We created a script that converted the json file into an image file of the same size as the original image with again, the path in white and all the rest in black. In total we have +-580 labeled images.

We created [Giusti flatten](scripts/giusti_flatten) to flatten the dataset into our
own format.

```{figure} media/giusti-example.jpg
Giusti dataset example
```


## Steentjes

Our last dataset, Steentjes, was a dataset with images from a forest in Schiplaken that we took ourselves. We created the masks for these images using the same tool as mentioned before. This dataset has 77 labeled images.

We also experimented with combining multiple datasets together. After some testing we got the best result from combining Freiburg and Steentjes together.

All these datasets were converted into TFRecords to work on the Google Cloud TPUs and were stored on Google Cloud Buckets to maximize training performance.

```{figure} media/steentjes-example.jpg
Steentjes dataset example
```
