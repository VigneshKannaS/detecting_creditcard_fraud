# Credit Card Fraud Detection

## Overview

This project focuses on detecting credit card fraud using various machine learning models. Given the highly imbalanced nature of the dataset, the project explores different approaches to handle outliers and model performance.

## Project Structure

1. **Dataset**: The dataset used for this project is highly imbalanced, containing a significant proportion of non-fraudulent transactions compared to fraudulent ones.
2. **Models**: Seven different models were developed to address the fraud detection problem, each with variations in handling outliers and model fitting.

## Model Versions

The project evaluates three versions of models with different handling of outliers:

1. **Model Undersampled with Outliers Present**: The model was trained with undersampled data that includes outliers. 
2. **Model Oversampled with Outliers Present**: The model was trained with oversampled data that includes outliers.
3. **Model Oversampled with Outlier Absent**: The model was trained with oversampled data that excludes outliers

### Final Approach

After analyzing the results, it was observed that outliers contain significant information about fraudulent transactions. Consequently, the following models were developed using the first two versions:

- **Logistic Regression 1**
- **Logistic Regression 2**
- **Decision Tree Classifier 1**
- **Decision Tree Classifier 2**
- **Support Vector Machine**
- **Random Forest Classifier 1**
- **Random Forest Classifier 2**

## Dataset

- **Source**: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]
- **Features**: [Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount]
- **Label**: 0 -> Legitimate Transaction, 1 -> Fraudulent Transaction

## Model Training and Evaluation

- **Metrics Used**: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Evaluation**: The models were evaluated using cross-validation and performance metrics to ensure robustness and reliability.

## How to Run

1. **Clone the Repository**
    ```bash
    git clone https://github.com/VigneshKannaS/Credit-Card-Fraud-Detection-Model.git
    cd Credit-Card-Fraud-Detection-Model
    ```

2. **Install Dependencies**
    Make sure to have Python installed and set up a virtual environment. Then, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Models**
    Execute the Python script to train and evaluate the models:
    ```bash
    python fraudDetection.py
    ```

## Results

The final models were evaluated based on their performance in detecting fraudulent transactions, with a focus on balancing precision and recall due to the imbalanced nature of the dataset.

## Contributions

- **Author**: Vignesh Kanna S
- **Date**: 1/07/2023

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Any references or acknowledgments]
