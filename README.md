# PCA Analysis on Breast Cancer Dataset

## Overview
This project is part of BAN6420 Milestone assignment to understand and apply Principal Component Analysis (PCA) using the breast cancer dataset from sklearn.datasets.

PCA is a technique used for dimensionality reduction, which helps in simplifying complex datasets while retaining the most important information. The goal of this project is to reduce the dataset to 2 principal components and visualize the results.

Additionally, as a bonus task, we will train a logistic regression model on the reduced dataset to predict whether a tumor is malignant or benign.



## Project Structure

Module_5_assignment/
│── pca_cancer_analysis.py                      # Main Python script for PCA and logistic regression
│── README.md                                   # Project documentation (this file)
│── requirements.txt                            # List of required Python libraries
│── visualization.png                           # PCA visualization output (generated after running script)
└── interpretation_of_the_pca_Visualisation     # Interpretation of the Visualization


## Understanding the Dataset
The dataset used in this project is the breast cancer dataset from sklearn.datasets. It contains 30 numerical features related to tumor characteristics, such as:

Mean radius
Mean texture
Mean perimeter
Mean area
Mean smoothness, etc.
The dataset also includes a target variable indicating the diagnosis:

0 → Malignant (cancerous)
1 → Benign (non-cancerous)


## Execute the code with below command:
  # python pca_cancer_analysis.py
# BAN6420_Milestone_Assignment_2
