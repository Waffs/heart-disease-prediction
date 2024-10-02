# heart-disease-prediction
This is my submission for Data Science Nigeria (DSN) AI Bootcamp Hackaton 2024, can be found in this [page](https://github.com/DataScienceNigeria/DSN-AI-Bootcamp-2024-Qualification-Project-Participation-and-Hackathon/blob/main/ML%20with%20Azure-Python.md)

## Overview
This project aims to develop a predictive model for heart disease using machine learning techniques. The dataset includes various health-related features, and the goal is to predict the presence or absence of heart disease.

## Dataset
The dataset consists of two files:
- `train_dataset.csv`: Contains the training data with features and the target variable (`target`).
- `test_dataset.csv`: Contains the test data with features but without the target variable.

## Columns
- **Id**: Unique identifier for each record.
- **Age**: Age of the individual.
- **Sex**: Gender of the individual (1 = male; 0 = female).
- **cp**: Chest pain type (0-3).
- **trestbps**: Resting blood pressure (in mm Hg).
- **chol**: Serum cholesterol in mg/dl.
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
- **restecg**: Resting electrocardiographic results (0-2).
- **thalach**: Maximum heart rate achieved.
- **exang**: Exercise induced angina (1 = yes; 0 = no).
- **oldpeak**: ST depression induced by exercise relative to rest.
- **slope**: Slope of the peak exercise ST segment (0-2).
- **ca**: Number of major vessels (0-3) colored by fluoroscopy.
- **thal**: Thalassemia (0-3).
- **target**: Presence of heart disease (1 = presence; 0 = absence) - only in the training dataset.

## Installation
To run this project, ensure you have the following Python packages installed:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib

## Usage
1. Load the datasets using Pandas.
2. Preprocess the data by encoding categorical variables.
3. Train a machine learning model (e.g., Random Forest).
4. Make predictions on the test dataset.
5. Generate a submission file containing the ID and predicted Target.

## Example Code
Here is a simplified version of the code used in this project:
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

## Load datasets
train_dataset = pd.read_csv('path_to_train_dataset.csv')
test_dataset = pd.read_csv('path_to_test_dataset.csv')

## Encode categorical variables
 ... (encoding steps)

## Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

## Make predictions
predictions = model.predict(X_test)

## Prepare submission file
submission_df = pd.DataFrame({'ID': test_dataset['Id'], 'Target': predictions})
submission_df.to_csv('submission_file.csv', index=False)

## Performance Metrics
The model's performance is evaluated using various metrics, including:

Accuracy
Precision
Recall
F1 Score
ROC-AUC Score

## Biases and Limitations
### Data Bias: The model's predictions are only as good as the data it was trained on. If the training dataset does not represent the general population, the model may not perform well in real-world scenarios.
### Feature Limitations: The features used in the dataset may not capture all relevant factors influencing heart disease. Important variables such as lifestyle factors, family history, and genetic predispositions may be missing.
### Overfitting: The model may perform exceptionally well on the training data but fail to generalize to unseen data if not properly validated.
### Interpretability: While Random Forest provides some insights into feature importance, it is still less interpretable than simpler models like logistic regression. This can make it challenging to understand the underlying reasons for predictions.
### Ethical Considerations: Misclassification could have serious implications for patients. False negatives may lead to undiagnosed conditions, while false positives can cause unnecessary anxiety and further medical tests.

## Conclusion
This project demonstrates the application of machine learning techniques to predict heart disease. Further feature engineering and hyperparameter tuning can improve the model's effectiveness.
