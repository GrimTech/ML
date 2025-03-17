import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from imblearn.over_sampling import SMOTE
import joblib

# 1. Generate Synthetic Data (Replace with real data)
np.random.seed(42)
n_samples = 10000
data = pd.DataFrame({
    'income': np.random.normal(60000, 20000, n_samples),
    'credit_score': np.random.normal(650, 50, n_samples),
    'debt': np.random.normal(15000, 5000, n_samples),
    'employment_years': np.random.uniform(0, 20, n_samples),
    'age': np.random.uniform(18, 70, n_samples),
    'employment': np.random.choice(['Full-DateTime', 'Part-DateTime', 'Unemployed'], n_samples),
    'marital_status': np.random.choice(['Married', 'Single', 'Divorced'], n_samples),
    'loan_status': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),  # 70% eligible
})
data['loan_amount'] = np.where(data['loan_status'] == 1, 
                               np.random.uniform(500, 50000, n_samples), 0)
data['dti'] = data['debt'] / data['income']  # Debt-to-Income Ratio

# 2. Data Cleaning
def clean_data(df):
    # Handle missing values
    for col in ['income', 'credit_score', 'debt', 'dti', 'employment_years', 'age']:
        df[col] = df[col].fillna(df[col].median())
    df['employment'] = df['employment'].fillna('Unknown')
    df['marital_status'] = df['marital_status'].fillna('Unknown')
    
    # Remove outliers (IQR method)
    for col in ['income', 'credit_score', 'debt', 'dti']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR))]
    
    # Fix data types
    df[['income', 'credit_score', 'debt', 'dti', 'employment_years', 'age']] = \
        df[['income', 'credit_score', 'debt', 'dti', 'employment_years', 'age']].apply(pd.to_numeric, errors='coerce')
    return df

data = clean_data(data)

# 3. Feature Engineering and Encoding
data['stable_income'] = (data['employment_years'] > 2).astype(int)
data = pd.get_dummies(data, columns=['employment', 'marital_status'], drop_first=True)

# 4. Data Splitting
X = data.drop(['loan_status', 'loan_amount'], axis=1)
y_class = data['loan_status']
y_reg = data['loan_amount']

X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# 5. Handle Imbalance with SMOTE (Classification)
smote = SMOTE(random_state=42)
X_train_res, y_class_train_res = smote.fit_resample(X_train, y_class_train)

# 6. Preprocessing Pipeline
num_features = ['income', 'credit_score', 'debt', 'dti', 'employment_years', 'age']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', 'passthrough', [col for col in X.columns if col not in num_features])
    ])

# 7. Classification Model (Eligibility)
class_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42))
])

# Train classification model
class_pipeline.fit(X_train_res, y_class_train_res)

# 8. Regression Model (Loan Amount)
reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
])

# Train regression model on eligible users only
train_eligible = X_train[y_class_train == 1]
y_reg_eligible = y_reg_train[y_class_train == 1]
reg_pipeline.fit(train_eligible, y_reg_eligible)

# 9. Evaluation
# Classification
y_class_pred = class_pipeline.predict(X_test)
print("Classification Metrics:")
print(f"Accuracy: {accuracy_score(y_class_test, y_class_pred):.3f}")
print(f"Precision: {precision_score(y_class_test, y_class_pred):.3f}")
print(f"Recall: {recall_score(y_class_test, y_class_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_class_test, class_pipeline.predict_proba(X_test)[:, 1]):.3f}")

# Regression
test_eligible = X_test[y_class_test == 1]
y_reg_pred = reg_pipeline.predict(test_eligible)
print("\nRegression Metrics:")
print(f"MAE: {mean_absolute_error(y_reg_test[y_class_test == 1], y_reg_pred):.2f}")
print(f"RMSE: {mean_squared_error(y_reg_test[y_class_test == 1], y_reg_pred, squared=False):.2f}")

# 10. Prediction Function
def predict_loan_eligibility(user_data):
    """
    Predicts loan eligibility and suitable amount.
    user_data: dict with keys: income, credit_score, debt, employment_years, age, 
               employment, marital_status
    """
    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])
    
    # Feature engineering
    user_df['dti'] = user_df['debt'] / user_df['income']
    user_df['stable_income'] = (user_df['employment_years'] > 2).astype(int)
    user_df = pd.get_dummies(user_df, columns=['employment', 'marital_status'], drop_first=True)
    
    # Align columns with training data
    for col in X.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[X.columns]  # Ensure same order
    
    # Predict eligibility
    eligible = class_pipeline.predict(user_df)[0]
    min_loan = 500  # Minimum loan amount
    
    if not eligible:
        return f"Not eligible for minimum loan amount (${min_loan})"
    
    # Predict loan amount
    amount = reg_pipeline.predict(user_df)[0]
    amount = max(min_loan, min(amount, 50000))  # Cap between $500 and $50,000
    return f"Eligible for loan amount: ${amount:.2f}"

# 11. Save Models
joblib.dump(class_pipeline, 'class_model.pkl')
joblib.dump(reg_pipeline, 'reg_model.pkl')

# Example Usage
user = {
    'income': 50000, 'credit_score': 700, 'debt': 10000, 
    'employment_years': 5, 'age': 30, 
    'employment': 'Full-Time', 'marital_status': 'Married'
}
print(predict_loan_eligibility(user))

# Optional: Load Models
# class_pipeline = joblib.load('class_model.pkl')
# reg_pipeline = joblib.load('reg_model.pkl')