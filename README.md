# Predictive Modeling for Vaccination

Predicts the probability of individuals receiving XYZ and seasonal flu vaccines using survey data. Built for the CNA Hackathon dataset.

## Problem

Two binary classification targets, evaluated on mean ROC AUC:
- **xyz_vaccine** — probability the respondent received the XYZ vaccine
- **seasonal_vaccine** — probability the respondent received the seasonal flu vaccine

## Dataset

| File | Description |
|------|-------------|
| **training_set_features.csv** | Survey responses for 26,707 respondents |
| **training_set_labels.csv** | Vaccine uptake labels |
| **test_set_features.csv** | Held-out respondents for submission |

Key features include demographics (age, sex, education, race), behavioral indicators (doctor recommendations, health opinions), and socioeconomic attributes.

## Approach

**Preprocessing**
- Age group strings converted to numeric midpoints
- Education encoded ordinally (1–4)
- `income_poverty` and `health_insurance` dropped due to >20% missingness
- Categorical columns (`marital_status`, `rent_or_own`, `employment_status`, `census_msa`, `race`, `sex`) one-hot encoded
- Remaining numeric nulls filled via random draw from observed values
- `doctor_recc_xyz` and `doctor_recc_seasonal` imputed using a logistic regression trained on all other features

A single `preprocess_df()` function is applied consistently to both train and test sets to prevent data leakage.

**Models trained and compared via 5-fold cross-validation**
- Logistic Regression (with StandardScaler)
- Random Forest
- Gradient Boosting

Best model selected by mean ROC AUC across both targets. GridSearchCV is used for hyperparameter tuning.

## Results

| Model | XYZ AUC | Seasonal AUC |
|-------|---------|--------------|
| Logistic Regression | ~0.83 | ~0.85 |
| Random Forest | ~0.85 | ~0.87 |
| Gradient Boosting | ~0.86 | ~0.88 |

*Scores are approximate and will vary by run due to random imputation.*

## Files

```
vaccine.ipynb       - Main notebook (preprocessing, EDA, training, submission)
vaccination.csv     - Sample submission output
README.md           - This file
```

## Requirements

```
pandas
numpy
scikit-learn
seaborn
matplotlib
```

## Usage

1. Upload the CNA Hackathon dataset to `/kaggle/input/cna-hackathon/`
2. Run all cells in `vaccine.ipynb`
3. `submission.csv` will be saved to the working directory
