# Loan Risk Assessment Model

This project is a machine learning solution for a money lending company to evaluate loan eligibility and recommend suitable loan amounts based on user risk profiles. It employs a two-step approach: a classification model to determine if a user qualifies for the minimum loan amount ($500), and a regression model to predict the optimal loan amount for eligible users (up to $50,000).

## Features
- **Eligibility Prediction**: Binary classification (Yes/No) for minimum loan eligibility.
- **Loan Amount Prediction**: Regression to estimate a suitable loan amount for eligible users.
- **Robust Data Pipeline**: Cleaning, feature engineering (e.g., debt-to-income ratio), and imbalance handling.
- **Optimized Models**: Random Forest for classification, XGBoost for regression—balanced for performance and accuracy.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **Libraries**:
  - `pandas`: Data handling and manipulation.
  - `numpy`: Numerical computations.
  - `scikit-learn`: Machine learning utilities and pipelines.
  - `xgboost`: Advanced regression modeling.
  - `imblearn`: SMOTE for handling imbalanced data.
  - `joblib`: Model saving and loading.

## Installation
1. **Clone or Download**: Obtain the project files.
   ```bash
   git clone <repository-url>
   cd loan-risk-assessment
Install Dependencies: Use pip to install required packages.
bash
pip install pandas numpy scikit-learn xgboost imblearn joblib
Verify: Ensure all libraries are installed.
bash
python -c "import pandas, numpy, sklearn, xgboost, imblearn, joblib"
Usage
Prepare Data: Replace the synthetic data in loan_model.py with your dataset (e.g., CSV file).
Expected columns: income, credit_score, debt, employment_years, age, employment, marital_status, loan_status, loan_amount.
Example: data = pd.read_csv('your_data.csv').
Run the Script: Execute the model training and evaluation.
bash
python loan_model.py
Outputs:
Classification metrics (Accuracy, Precision, Recall, ROC-AUC).
Regression metrics (MAE, RMSE).
Example prediction for a sample user.
Make Predictions: Use the predict_loan_eligibility function.
python
user = {
    'income': 50000, 'credit_score': 700, 'debt': 10000, 
    'employment_years': 5, 'age': 30, 
    'employment': 'Full-Time', 'marital_status': 'Married'
}
result = predict_loan_eligibility(user)
print(result)
Example Output: "Eligible for loan amount: $12345.67" or "Not eligible for minimum loan amount ($500)".
Saved Models: Models are saved as class_model.pkl (classification) and reg_model.pkl (regression) for reuse.
Load them with:
python
import joblib
class_pipeline = joblib.load('class_model.pkl')
reg_pipeline = joblib.load('reg_model.pkl')
Project Structure
loan_model.py: Main script containing data generation, cleaning, training, and prediction logic.
class_model.pkl: Trained classification model (generated after running).
reg_model.pkl: Trained regression model (generated after running).
How It Works
Data Cleaning: Handles missing values (median/mode imputation), removes outliers (IQR method), and fixes data types.
Feature Engineering: Adds dti (debt-to-income ratio) and stable_income (employment > 2 years).
Preprocessing: Scales numerical features, one-hot encodes categorical variables.
Classification: Random Forest predicts eligibility with SMOTE for balanced classes.
Regression: XGBoost estimates loan amounts for eligible users, capped at $500-$50,000.
Evaluation: Assesses performance with standard metrics (ROC-AUC for classification, MAE/RMSE for regression).
Customization
Real Data: Replace synthetic data with your dataset in loan_model.py.
Features: Add columns like credit_history_length or loan_purpose as needed.
Hyperparameters: Tune n_estimators, max_depth, etc., for better performance (e.g., via GridSearchCV).
Deployment: Extend to a Flask API:
python
from flask import Flask, request, jsonify
app = Flask(__name__)
class_model = joblib.load('class_model.pkl')
reg_model = joblib.load('reg_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = predict_loan_eligibility(data)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
Notes
Synthetic Data: The included dataset is for demonstration. Use real data for production.
Permissions: No special OS permissions required beyond Python execution.
Testing: Run in a virtual environment or VM for safety:
bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
Compliance: Ensure fairness and document predictions for regulatory purposes if used commercially.
Scalability: Suitable for small-to-medium datasets (<100k rows); scale with dask for larger data.
Contributing
Submit issues or pull requests for enhancements (e.g., additional features, model improvements).
License
This project is for educational purposes and not licensed for commercial use without modification and compliance with lending regulations.

---
