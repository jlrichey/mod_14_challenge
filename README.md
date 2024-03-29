# Module 14 Challenge - Machine Learning Trading Bot

<img src="images/challenge14_header_img.png" alt="drawing" width="800"/>

## Overview

For this challenge I am developing an algorithmic trading bot to enhance the existing trading signals, currently utilized by a top financial advisory firm, with machine learning algorithms. This will give the firm a competitive advantage in a highly dynamic industry where the ability to adapt to new data is paramount. 

## Dataset

The data was provided in `csv` format with over 4,300 rows of MSCI-based ("Morgan Stanley Capital International") investment data for emerging markets.

The dataset includes the following OHLCV fields:
* date
* open
* high
* low
* close
* volume

## Libraries and Dependencies

The [notebook](machine_learning_trading_bot.ipynb) loads the following libraries and dependencies.

```python
# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")
```

## Preprocessing of the Data

Several features were dropped (EIN and NAME) due to the irrelevance of the data to the deep learning model goals. 

`OneHotEncoder` from `scikit-learn` was used to convert the categorical data (identified using `.dtypes`) to binary numerical values for the model and placed into a dataframe which was concatenated with the original dataframe's numerical values.

The features (X) and target (y) were split into training and testing datasets using the `train_test_split` function from the `scikit-learn` machine learning library. The datasets were then scaled utilizing `StandardScaler()`.

## Binary Classification Models using a Neural Network

Three distinct deep learning models using a neural network were compiled for analysis and evaluation using TensorFlow Keras. The following are the details and results of each binary classification model. 

### Support Vector Machine (SVM) Model (scv classifier)

| Plot            | Classification Report            |
|--------------------|--------------------|
| ![Image1](images/SVM_model_plot.png) | ![Image2](images/LR_model_class1.png) |

### Model Tuning 1 - Training Window Adjustments

| 1 Month            | 6 Month            |
|--------------------|--------------------|
| ![Image1](images/tuned_1mth_training_window_plot.png) | ![Image2](images/tuned_6mth_training_window_plot.png) |
| ![Image1](images/tuned_1mth_training_window_class.png) | ![Image2](images/tuned_6mth_training_window_class.png) |


### Model Tuning 2 - Short SMA (Simple Moving Average) Input Features

| 2 Days*            | 10 Days*            |
|--------------------|--------------------|
| ![Image1](images/tuned_SMA_short_2d_plot.png) | ![Image2](images/tuned_SMA_short_10d_plot.png) |
| ![Image1](images/tuned_SMA_short_2d_class.png) | ![Image2](images/tuned_SMA_short_10d_class.png) |

*Note: "Days" are actually 15 minute periods.*

### Model Tuning 3 - Long SMA (Simple Moving Average) Input Features

| 50 Days*            | 200 Days*            |
|--------------------|--------------------|
| ![Image1](images/tuning_SMA_long_50d_plot.png) | ![Image2](images/tuning_SMA_long_200d_plot.png) |
| ![Image1](images/tuning_SMA_long_50d_class.png) | ![Image2](images/tuning_SMA_long_200d_class.png) |

*Note: "Days" are actually 15 minute periods.*

### Logistic Regression (LR) Model


| Plot            | Classification Report            |
|--------------------|--------------------|
| ![Image1](images/LR_model_plot.png) | ![Image2](images/LR_model_class1.png) |

## Summary

All three models had an approximate accuracy of 0.73 and an approximate loss of 0.55 (A2 was 0.56) when rounded to the nearest hundredth. The original model was the least complex with two hidden layers and one output layer. Its simplicity is seen in the step detail performance of 0 to 658us (microseconds). I would recommend the original model for these reasons. 

If I were to investigate further models for optimization, I would experiment with fewer hidden layers, fewer nodes, and more and less epochs, among other approaches. 

## Sources

The following sources were consulted in the completion of this project. 

* [pandas.Pydata.org API Reference](https://pandas.pydata.org/docs/reference/index.html)
* [Tensorflow Keras documentation](https://www.tensorflow.org/guide/keras)
* [scikit-learn documentation](https://scikit-learn.org/stable/)
* UCB FinTech Bootcamp instructor-led coding exercises
* ChatGPT for LeakyReLU integration syntax