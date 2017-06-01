## Glacier Detect ML
Teaching a computer to identify a glacier in satellite imagery

### Introduction
This set of python scripts can be used to train a simple logistic regression model to identify a glacier in Landsat satellite imagery. The idea is to assign a set of "features" to each pixel in the image (intensity in each channel, etc.) and then train a linear model to categorize each pixel as glacier or something not glacier, like open water, land, clouds, etc.

The ability to automate detection of glaciers in satellite imagery would help geo scientists who study glacier evolution, movement and interaction with surrounding land and water.

### Requirements
  * Python 3.5
  * [Tensor Flow](https://www.tensorflow.org/)   
  * [Rasterio](https://github.com/mapbox/rasterio)
  * Numpy
  * Matplotlib

### Usage
Before training, you need to construct the feature and label vectors for training and testing.

      $ (training_set, training_labels, test_set, test_labels) = features.construct_features()

The `construct_features()` function uses a `CONFIG` variable to locate the Landsat images, bounding boxes for glacier and non-glacier regions, etc.

**TODO**

1. Provide configuration as an argument to `construct_features()`
2. Switch to Amazon AWS for sourcing Landsat imagery

If you would like to save the features for later training:

      $ features.save_features(training_set, training_labels, test_set, test_labels, 'path/to/output')

To run the training:

      $ python train.py

**TODO**

1. refactor `train.py` so that there are no hard-coded paths and values.

You can visualize training metrics using [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) by running:

      $ tensorboard --logdir=path/to/logs
