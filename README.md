# Machine Learning Ensemble Methods

This repository contains Python code implementing three machine learning ensemble methods: Decision Trees, Random Forest, and AdaBoost. These methods are applied to the Titanic dataset for classification tasks.

## Decision Tree Classifier

- Implementation of a Decision Tree Classifier with options for specifying the maximum depth and splitting criterion (Gini impurity).
- Accuracy achieved on the Titanic dataset: XX.XX%

## Random Forest Classifier

- Implementation of a Random Forest Classifier using Decision Trees as base classifiers.
- Features include controlling the maximum depth, splitting criterion, and the number of trees in the forest.
- Accuracy achieved on the Titanic dataset: XX.XX%

## AdaBoost Classifier

- Implementation of an AdaBoost Classifier with Decision Trees as weak learners.
- The number of learners, learning rate, maximum depth, and splitting criterion are configurable.
- Accuracy achieved on the Titanic dataset: XX.XX%

### Dataset Preparation

The Titanic dataset is used for these experiments. It undergoes preprocessing steps, including one-hot encoding of categorical features.

### Usage

You can use the provided Python scripts to train and evaluate the models. Make sure you have the required libraries (NumPy, Pandas, Seaborn, Scikit-Learn) installed.

```bash
pip install numpy pandas seaborn scikit-learn
