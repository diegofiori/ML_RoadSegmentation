# ML_RoadSegmentation

README for the Machine Learning CS-433: Project 2 - Road Segmentation

Group members: 
- Diego Fiori
- Paolo Colusso 
- Valerio Volpe

CrowdAI team name: LaVolpeilFioreEilColosso

## Architecture

The files created and the functions developed are presented in the following sections:

* [Helpers](#helpers)
* [Preprocessing](#prepr)
* [Regression](#regression)
* [Neural Nets](#cnn)
* [Post Processing](#pp)
* [Submission](#subm)


### <a name="helpers"></a>Helpers
```helpers_img.py```

Contains the functions to load and read the data, perform basic procsseing of the images, compute the F1 score, and create the submission.

### <a name="prepr"></a>Preprocessing
```preprocessing.py```

Contains the function to pre-process the images. A series of functions are created to:
 * extend the dataset by means of rotation and flip
 * extend the borders of the image
 * apply filters on the images
 * add channels to the image
 * extract the features as mean and variance of the channels
 * take features of the polynomials
 
 ```dataset.py```

Class used to read the set of images.

### <a name="regression"></a>Logistic and Ridge Regression
```helpers_regression.py```

Tools to perform regression with cross validation. 

Contains the function used to:
 + split the data into train and test set
 + call the preprocessing functions
 + perform regression
 + call the post-processing functions
 
```Cross_Validation_regression.ipynb```

 + performs regularised logistic regression with cross-validation
 + performs ridge regression with cross-validation

### <a name="cnn"></a>Neural Nets
```NeuralNets.py```: contains the classes fot the *Simple Net*, the *U-Net* and the *Deep Net*.

```Bagging_Net.py```: contains the functions used to run the bootstrap-like neural net.

The following notebooks can be used to define and run the models:
+ Net with bootstrapping: ```Bagging_Net.ipynb```
+ U-Net:  ```U-Net.ipynb```,
+ Deep Net: ```RUN.ipynb```

```training_nn.py```: contains the functions to train neural networks.

 ```Models```: folder containing the the models created.

### <a name="pp"></a>Postprocessing
```Post_processing.py```

Contains the functions which perform post-processing operations on the predictions obtained for the images from either of the models mentioned above.

### <a name="subm"></a>Submission

```mask_to_submission.py```
```submission.py```
```submission_to_mask.py```

## CrowdAI results
Username: Paolo Colusso

Submission ID Number: 25160

## References

+ Statistical learning: James, Witten, Hastie, Tibshirani, *Introduction to Statistical Learning*, see [details](https://www-bcf.usc.edu/~gareth/ISL/).

+ Image processing: Burger, Burge, *Digital Image Processing. An Algorithmic Introduction Using Java*, see [details](https://www.springer.com/de/book/9781447166832).

+ U-Net: Ronneberger, O., Fischer, P., and Brox, T., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015.

+ Neural nets: EPFL course available [here](https://fleuret.org/ee559-2018/dlc/).
