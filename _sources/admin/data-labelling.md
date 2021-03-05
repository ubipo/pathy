# Data Labelling

To train our convolutional semantic image segmentation neural network 
we need labelled data of forest paths. Most data online however is 
meant for autnomous driving, usually on solid (asphalt/compacted gravel)
roads/paths.

## Freiburg

One (near) exception to this is the 
[Freiburg deepscene forest dataset](http://deepscene.cs.uni-freiburg.de/segment_random/FOREST_RGB). While this dataset still mostly contains 
compacted gravel park roads, it also contains a lot of smaller soft paths.

As such we adapted and used it for the training of our network.

## Giusti

Another excellent dataset of forest paths is the data set from Giusti et al.'s [A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots](https://people.idsia.ch/~guzzi/DataSet.html).

This data however is unfortunately only labelled as "left", "straight" or "center". 

## Own data

t.b.c.

## Crowd Sourced Labelling

To label the Giusti and our own data we employed the help of our 
fellow students in a crowd sourced labelling effort.

We set up a custom version of 
[Django Labeller](https://github.com/Britefury/django-labeller) 
on a Google Cloud VPS and used a 
[collaborative spreadsheet](https://docs.google.com/spreadsheets/d/1F_qDVuE1kOOzWPlyxjIwmBufMYoukAm_Qs4766otick) 
and a simple 
[instructional document](https://hackmd.io/eotmVGgfR6CO1zPifdQnyw)
to coordinate the labelling.

![Our Django Labeller instance](images/labeller.png)
*Our Django Labeller instance*

<br>

![Instructional document](images/document.png)
*Instructional document to coordinate labelling*
