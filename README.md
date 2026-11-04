Create README.md and paste:

markdown
# Loan Approval Prediction

Predicts whether a loan application should be approved based on applicant
details and financial information, using machine learning, with a Streamlit
app for live predictions.

## Problem Statement
Banks must decide quickly and consistently whether to approve a loan. This
project builds and compares ML classifiers that predict approval
(Approved / Rejected) from applicant and financial features.

## Objectives
1. Explore the data and test which factors are associated with approval
   (chi-square, Mann-Whitney U, Welch t-test, confidence intervals).
2. Build a baseline and three tuned ML models with leak-free preprocessing
   and cross-validation.
3. Compare models using precision, recall, F1, ROC-AUC and confusion matrix.
4. Deploy the best model as a Streamlit web app.

## Dataset
Loan Approval Prediction Dataset (Kaggle, uploaded by Archit Sharma).
4,269 applications, 12 features + target (`loan_status`).
File: `data/loan_approval_dataset.csv`

## Project Structure
```
data/        dataset
outputs/     figures and result tables
models/      saved trained model
```

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Results
To be added.

## Limitations
To be added.