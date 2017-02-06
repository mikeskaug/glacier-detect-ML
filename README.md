## Glacier Detect ML
Teaching a computer to identify a glacier in satellite imagery

### Introduction
This set of Jupyter notebooks can be used to train a simple linear model to identify a glacier in Landsat satellite imagery. The idea is to assign a set of "features" to each pixel in the image (intensity in each channel, position, etc.) and then train a linear model to categorize each pixel as glacier or something not glacier, like open water, land, clouds, etc.

The ability to automate detection of glaciers in satellite imagery would help geo scientists who study glacier evolution, movement and interaction with surrounding land and water.

### Requirements
  * Python 3.5
  * [Jupyter 4.1.0](http://jupyter.org/)
  * [Tensor Flow](https://www.tensorflow.org/)   
  * [Rasterio](https://github.com/mapbox/rasterio)
  * Numpy
  * Matplotlib

### Usage
The Landat imagery is available publicly from Amazon Web Services: [AWS Landsat](https://pages.awscloud.com/public-data-sets-landsat.html)

The notebooks are meant to be run in the following order:

  1. resize_image.ipynb

  2. construct_features.ipynb

  3. train_model.ipynb

  4. process_image.ipynb

Comments and explanations are included in the notebooks.

The notebooks assume that the following four images (each containing 7 spectral bands) have been downloaded locally in `./images/`

* LE70322482009163EDC00
* LE70322482009195EDC00
* LE70332482010173EDC00
* LE70342482009177EDC00

See the documentation on [AWS](https://pages.awscloud.com/public-data-sets-landsat.html) to parse the image names

### TODO

1. Use Landsat 8 imagery which is available on AWS and create a script to download and preprocess the images.

2. weight connectedness in the model
