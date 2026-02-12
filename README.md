# Telecom Customer Churn Prediction

Predicting customer churn in a telecommunications company using machine learning.  
Goal: Identify at-risk customers early so the retention team can intervene with targeted offers, reducing revenue loss.

![Churn Overview](https://via.placeholder.com/800x300.png?text=Customer+Churn+Dashboard)  
*(Replace with actual screenshot of model performance or churn dashboard)*

## Project Overview

This project builds interpretable classification models to predict whether a customer will churn (cancel service).  
We compare **Logistic Regression** (linear, highly interpretable) with **Decision Tree** (non-linear, better performance), including hyperparameter tuning.

**Key business motivation**:
- Churn rate in dataset: **~14.5%**
- Acquiring new customers is **5–25×** more expensive than retaining existing ones
- Reducing churn by ~4–5 percentage points can save **hundreds of millions to billions of KES** annually (depending on subscriber base and ARPU)

## Dataset

- Source: Synthetic / anonymized telecom churn dataset (~3,333 records)
- File: `bigml_59c28831336c6604c800002a.csv`
- Target: `churn` (binary: 0 = stay, 1 = churn)
- Important features:
  - `total day minutes` / `total day charge`
  - `customer service calls`
  - `international plan`
  - `voice mail plan`
  - `state`

## Project Structure

```text
telecom-churn-prediction/
├── data/
│   └── bigml_59c28831336c6604c800002a.csv          # original dataset
├── notebooks/
│   ├── 01_EDA.ipynb                               # Exploratory Data Analysis
│   ├── 02_Modeling_Baseline.ipynb                 # Simple models
│   ├── 03_Hyperparameter_Tuning.ipynb             # Grid & Randomized Search
│   └── 04_Final_Model_Comparison.ipynb            # Best model evaluation
├── src/
│   ├── data_preprocessing.py                      # (optional) reusable preprocessing
│   └── modeling.py                                # (optional) model training helpers
├── models/                                        # saved best models (.joblib / .pkl)
├── README.md
└── requirements.txt

## Overview
This project focuses on predicting customer churn using machine learning techniques. The dataset used is `bigml_59c28831336c6604c800002a.csv`, and the analysis is performed in the notebook `churn.ipynb`.

## Project Structure
  - `bigml_59c28831336c6604c800002a.csv`: Main dataset for churn prediction.
  - `churn.ipynb`: Jupyter notebook containing data exploration, preprocessing, modeling, and evaluation steps.

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn



## Project Steps
1. **Data Loading & Exploration**: Load the dataset and perform exploratory data analysis (EDA).
2. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale features as needed.
3. **Model Building**: Train machine learning models to predict churn.
4. **Evaluation**: Assess model performance using appropriate metrics.
5. **Conclusion**: Summarize findings and potential next steps.
