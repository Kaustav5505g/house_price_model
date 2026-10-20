import joblib
import pandas as pd

# 1. Load both saved models
linear_model = joblib.load('linear_regression_model.pkl')
logistic_model = joblib.load('logistic_regression_model.pkl')

# 2. Define a new house input (e.g., a 1,500 sqft apartment)
# We use a pandas DataFrame because scikit-learn expects feature column names
new_house = pd.DataFrame([[1500]], columns=['Area'])

# 3. Get Predictions from both models
predicted_price = linear_model.predict(new_house)[0]
luxury_probability = logistic_model.predict_proba(new_house)[0][1] # Probability of being class 1 (Luxury)
luxury_prediction = logistic_model.predict(new_house)[0]

# 4. Print out the results
print("==========================================")
print(f"🏠 Input House Area: {new_house['Area'].values[0]} sqft")
print("==========================================")
print(f"📈 Linear Regression Prediction:")
print(f"   Estimated Price: ₹{predicted_price:,.2f}")
print("------------------------------------------")
print(f"🏷️ Logistic Regression Prediction:")
print(f"   Category: {'Luxury' if luxury_prediction == 1 else 'Budget'}")
print(f"   Confidence / Luxury Probability: {luxury_probability * 100:.1f}%")
print("==========================================")