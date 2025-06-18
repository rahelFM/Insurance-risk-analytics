# predict.py
import joblib
model = joblib.load("models/insurance_model.pkl")
def predict_charge(age, bmi, smoker, children):
    smoker_code = 1 if smoker.lower() == 'yes' else 0
    return model.predict([[age, bmi, smoker_code, bmi*smoker_code, children]])[0]