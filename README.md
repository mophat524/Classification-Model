# SyriaTel Customer Churn Prediction

Predicting customer churn in a telecommunications company using machine learning.  
Goal: Identify at-risk customers early so the retention team can intervene with targeted offers, reducing revenue loss.

<img width="846" height="482" alt="image" src="https://github.com/user-attachments/assets/6efe355e-4068-4c53-8569-5d27b22c77bc" />


## Project Overview

This project builds interpretable classification models to predict whether a customer will churn (cancel service).  
We compare **Logistic Regression** (linear, highly interpretable) with **Decision Tree** (non-linear, better performance), including hyperparameter tuning.

**Key business motivation**:
- Churn rate in dataset: **~14.5%**
- Acquiring new customers is **5–25×** more expensive than retaining existing ones
- Reducing churn by ~4–5 percentage points can save money annually (depending on subscriber base and ARPU(Average Revenue Per User))

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
