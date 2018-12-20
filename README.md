# ML_RoadSegmentation

README for the Machine Learning CS-433: Project 2 - Road Segmentation

Group members: 
- Diego Fiori
- Paolo Colusso 
- Valerio Volpe

CrowdAI team name: LaVolpeilFioreEilColosso

## Architecture

A data folder with the following structure must be created:

```
training/
test_set_images/
```

The files created and the functions developed are presented in the following sections:

* [Helpers](#helpers)
* [Preprocessing](#prepr)
* [Logistic Regression](#logistic)
* [Ridge Regression](#ridge)
* [Neural Nets](#cnn)
* [Post Processing](#pp)

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

### <a name="logistic"></a>Regression
```helpers_regression.py```

Tools to perform regression with cross validation. Contains the function used to:
 + split the data into train and test set
 + preprocess the data
 + perform regression
 + post-process the data
 
```Cross_Validation_regression.ipynb```

Performs regularised logistic regression by cross-validation on the regularising parameter.

Computes the cross-validation F1.

### <a name="ridge"></a>Ridge Regression
```cv_ridge.py```

Performs Ridge regression by cross-validation on the regularising paramete and on the threshold which determines whether the prediction is set to 0 or 1.

Computes the cross-validation F1.

### <a name="cnn"></a>Neural Nets

### <a name="pp"></a>Postprocessing
```Post_processing.py```

Contains the functions which perform post-processing operations on the predictions obtained for the images from either of the models mentioned above.

## References

+ Statistical learning: James, Witten, Hastie, Tibshirani, *Introduction to Statistical Learning*, see [details](https://www-bcf.usc.edu/~gareth/ISL/).

+ Image processing: Burger, Burge, *Digital Image Processing. An Algorithmic Introduction Using Java*, see [details](https://www.springer.com/de/book/9781447166832).

+ Neural nets: EPFL course available [here](https://fleuret.org/ee559-2018/dlc/).
