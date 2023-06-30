---
permalink: /cookbook/
title: Cookbook

layout: archive
author_profile: true
---

Personal reference scripts for commonly used code

# Cookbook
Personal reference scripts for commonly used code

- **[Machine Learning](https://github.com/MacalusoJeff/Cookbook/tree/main/MachineLearning):** A folder containing scripts for commonly used machine learning code
    - **[Preprocessing.py](https://github.com/MacalusoJeff/Cookbook/blob/main/MachineLearning/Preprocessing.py):** Preparing data for machine learning tasks, primarily using pandas and sklearn
    - **[scikit-learn](https://github.com/MacalusoJeff/Cookbook/tree/main/MachineLearning/scikit-learn):** Also includes LightGBM and XGBoost
        - [ModelTraining.py](https://github.com/MacalusoJeff/Cookbook/blob/main/MachineLearning/scikit-learn/ModelTraining.py): Cross validation, hyperparameter tuning, feature selection, etc.
        - [Evaluation.py](https://github.com/MacalusoJeff/Cookbook/blob/main/MachineLearning/scikit-learn/Evaluation.py): Evaluation plots, collecting eval metrics, learning curves, feature importance, etc.
        - [LighTGBM.py](https://github.com/MacalusoJeff/Cookbook/blob/main/MachineLearning/scikit-learn/LightGBM.py): Early stopping and other code that's convenient to copy/paste
    - **[TensorFlow](https://github.com/MacalusoJeff/Cookbook/tree/main/MachineLearning/TensorFlow)**
        - [Keras.py](https://github.com/MacalusoJeff/Cookbook/blob/main/MachineLearning/TensorFlow/Keras.py): Commonly used code for Keras
        - [KerasMNIST.py](https://github.com/MacalusoJeff/Cookbook/blob/main/MachineLearning/TensorFlow/KerasMNIST.py): Training a convolutional net on the MNIST data with Keras
        - [TensorFlowMNIST.py](https://github.com/MacalusoJeff/Cookbook/blob/main/MachineLearning/TensorFlow/TensorFlowMNIST.py): Training a convolutional net on the MNIST data with TensorFlow
    - **[PyTorch](https://github.com/MacalusoJeff/Cookbook/tree/main/MachineLearning/PyTorch)**
        - [PyTorch.py](https://github.com/MacalusoJeff/Cookbook/blob/main/MachineLearning/PyTorch/PyTorch.py): Commonly used code for PyTorch
        - [PyTorchMNIST.py](https://github.com/MacalusoJeff/Cookbook/blob/main/MachineLearning/PyTorch/PyTorchMNIST.py): Training a convolutional net on the MNIST data with PyTorch
    - **[SparkML](https://github.com/MacalusoJeff/Cookbook/tree/main/MachineLearning/SparkML)**
        - [SparkML.py](https://github.com/MacalusoJeff/Cookbook/blob/main/MachineLearning/SparkML/SparkML.py): Commonly used code for SparkML. Includes preprocessing, hyperparameter tuning, cross validation, and so on.
- **[Plotting](https://github.com/MacalusoJeff/Cookbook/tree/main/Plotting):** Code snippets for common plots
    - **[Matplotlib.py](https://github.com/MacalusoJeff/Cookbook/tree/main/Plotting/Matplotlib.py)**
    - **[Plotly.py](https://github.com/MacalusoJeff/Cookbook/tree/main/PlottingPlotly.py)**
- **[Misc](https://github.com/MacalusoJeff/Cookbook/tree/main/Misc):** For scripts that don't fit within any other folders
    - **[EDA.py](https://github.com/MacalusoJeff/Cookbook/tree/main/Misc/EDA.py):** EDA reports, missing values, and outliers
    - **[NLP.py](https://github.com/MacalusoJeff/Cookbook/tree/main/Misc/NLP.py):** NLTK natural language processing tasks
    - **[PySpark.py](https://github.com/MacalusoJeff/Cookbook/tree/main/Misc/PySpark.py):** Missing values, datatype conversions, encoding categorical columns, and prepping data for models
- **[DevOps](https://github.com/MacalusoJeff/Cookbook/tree/main/DevOps)**: A folder containing scripts for operationalizing machine learning models
    - **[Flask](https://github.com/MacalusoJeff/Cookbook/tree/main/DevOps/Flask)**: Operationalizing a trained machine learning model as a RESTful API
        - [Web App](https://github.com/MacalusoJeff/Cookbook/blob/main/DevOps/Flask/app.py)
        - [Sending Requests](https://github.com/MacalusoJeff/Cookbook/blob/main/DevOps/Flask/request.py)
