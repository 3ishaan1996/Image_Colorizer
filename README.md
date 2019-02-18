# Image Colorizer

This Project aims to convert a grayscale image to colorized image, using Linear regression and Logistic regression models.

## Getting Started

### Prerequisites

```
pip install numpy
pip install opencv-python
```

## Running the tests

```
python colorizer.py
```
Place the input file in the desired directory, and add it as a variable to the colorizer.py file.

## Representing the process

The image is represented as a 2D matrix of pixels values. We are mapping the grayscale to RGB values, but instead of having the RGB value for a single pixel, we consider the RGB value for a block of 3X3 surrounding pixels.

## Data

We are collecting the training images of flowers from http://pexels.com/. For every pixel, we include the RGB values of all the pixels in the surrounding grid of 3X3. We scale the RGB values from 0-255 to 0-1. This is done to ensure that our gradient descent function doesn't overflow.

We perform quantization so that complex images can be decomposed into a palette of colors (6 to 10). Since humans are not able to visualize every possible minute color combination, quantization can be done in such a way so that there is minimal perceptual loss, while making training and classification easier. This step is done using K-means clustering algorithm.

## Evaluating the Model

![Alt text](loss_function.png?raw=true "Loss Function")

While colorizing a grayscale image, the program makes some perceptual errors. These errors include failure to capture long-range consistency, frequent confusions between red and blue, and a default sepia tone on some images.

We compute the final loss at the end to compare the image we colorize to the original image. Also, the loss function depends on the image size - higher the image size; more the pixels leading to more training data and ultimately higher loss. Relative evaluation between images of different sizes is difficult as a result.

## Training the models

Training the model on a set of images becomes time consuming especially for higher resolution images. Generally, an image will contain far fewer colors than the entire set of color possibilities. One approach to reducing the color space is to assign colors into a set of predefined bins of color ranges. This serves to effectively reduce the color space and complexity of the image. We expand on this idea by using k-means to reduce the color space of K representative colors.

For linear regression, convergence was achieved when the difference in loss of the current iteration is less than 0.05*(previous loss). Two methods were implemented to avoid over-fitting: normalizing the data and including a regularization term in the loss function.

For classification, instead of creating classes based on bins on each R(0-255), G(0-255) and B(0-255) value, we use image quantization to create unique colors to classify on. This technique is much more helpful, as we get a palette of colors based on our image data rather.  Couple of problems we might face here; we need to have palette of most of the general colors. We took images that have a variety of colors on it to ensure this.

After setting the number of unique colors, we use logistic regression with sigmoid function as one-vs-many classifier to classify each pixel as one of the colors of the palette. Logistic regression adds one layer of non-linearity to our model.

## Sample Results

Original image

![Alt text](2.jpg?raw=true "Original Image")

Colorized image

![Alt text](2_c.png?raw=true "Colorized Image")


## Acknowledgments

* Kunal Shah
* Vedang Mehta
* Nick Romanov
