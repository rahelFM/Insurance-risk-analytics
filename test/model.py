import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os
import joblib

# 1. Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Goes up two levels from test/
data_path = os.path.join(project_root, 'Insurance-risk-analytics\data\Statistical_modeling (1)\SM\data', 'insurance.csv')
model_dir = os.path.join(project_root, 'Insurance-risk-analytics\models')
model_path = os.path.join(model_dir, 'insurance_model.pkl')

# 2. Load Data
try:
    df = pd.read_csv(data_path)
    print("Data loaded successfully. Shape:", df.shape)
except Exception as e:
    print("ERROR: Data loading failed -", str(e))
    print("Tried path:", data_path)
    exit(1)

# 3. Feature Engineering
df['smoker_code'] = df['smoker'].map({'yes': 1, 'no': 0})
df['smoker_bmi'] = df['smoker_code'] * df['bmi']
df['age_squared'] = df['age'] ** 2

# 4. Model Training
X = df[['age', 'bmi', 'smoker_code', 'smoker_bmi', 'children']]
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
print("\nModel Performance:")
print(f"- R² Score: {r2_score(y_test, predictions):.3f}")
print(f"- MAE: ${mean_absolute_error(y_test, predictions):,.0f}")

# 6. Save Model (with directory creation)
os.makedirs(model_dir, exist_ok=True)  # Creates 'models' folder if missing
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")

# 7. Verify the saved model
try:
    loaded_model = joblib.load(model_path)
    print("Model verification: Loaded successfully!")
except Exception as e:
    print("Model verification failed:", str(e))
    # Add to model.py after training
import shap

# Explain model predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Generate visualization
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
plt.savefig('C:\\Users\\ahel\\Desktop\\KAIM 5-6\\Week 3\\Insurance-risk-analytics\\models\\reports\\shap_summary.png')

# Individual prediction explanation
sample_idx = 0
shap.force_plot(explainer.expected_value, shap_values[sample_idx], 
                X_test.iloc[sample_idx], feature_names=X.columns,
                matplotlib=True, show=False)
plt.savefig(f'C:\\Users\\ahel\\Desktop\\KAIM 5-6\\Week 3\\Insurance-risk-analytics\\models\\reports\\shap_force_plot_{sample_idx}.png')