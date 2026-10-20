import os
import glob
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Automatically download dataset via kagglehub
path = kagglehub.dataset_download("arvanshul/gurgaon-real-estate-99acres-com")
csv_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.read_csv(csv_files[0], low_memory=False)

print(f"Total rows loaded: {len(df)}")

# 2. Helper function to clean text/units and convert to float
def clean_val(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower().replace('sqft', '').replace('sq.ft', '').replace(',', '').strip()
    try:
        return float(val_str)
    except:
        return None

# Clean area and price columns
for col in ['SUPER_SQFT', 'BUILTUP_SQFT', 'AREA']:
    if col in df.columns:
        df[col] = df[col].apply(clean_val)

df['Area'] = df['SUPER_SQFT'].fillna(df['BUILTUP_SQFT']).fillna(df['AREA'])
df['Amount'] = df['PRICE'].apply(clean_val)

# Drop missing values
df = df.dropna(subset=['Area', 'Amount'])

# Filter realistic ranges based on our previous summary
df = df[(df['Area'] > 100) & (df['Area'] < 10000) & (df['Amount'] > 1000) & (df['Amount'] < 200000)]
print(f"Rows remaining after filtering: {len(df)}")

if len(df) > 10:
    # 3. Create Classification Target: Is it a Luxury house? (1 if above median price, 0 otherwise)
    median_price = df['Amount'].median()
    df['Is_Luxury'] = (df['Amount'] > median_price).astype(int)

    X = df[['Area']]
    y = df['Is_Luxury']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Logistic Regression Classifier
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 5. Print Evaluation Metrics
    print(f"\n--- Classification Model Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. --- MATPLOTLIB VISUALIZATION (Sigmoid Curve) ---
    plt.figure(figsize=(10, 6))
    
    # Scatter actual test points (0 or 1)
    plt.scatter(X_test, y_test, color='blue', alpha=0.4, label='Actual Classes (0: Budget, 1: Luxury)', zorder=3)

    # Generate a smooth curve for the Logistic Regression probability (Sigmoid curve)
    X_range = np.linspace(X['Area'].min(), X['Area'].max(), 300).reshape(-1, 1)
    y_prob = model.predict_proba(X_range)[:, 1] # Probability of being class 1 (Luxury)

    plt.plot(X_range, y_prob, color='red', linewidth=2.5, label='Logistic Regression Sigmoid Curve', zorder=4)
    plt.axhline(0.5, color='gray', linestyle='--', label='Decision Threshold (0.5)')

    plt.xlabel('Area (sqft)', fontsize=12)
    plt.ylabel('Probability of Being Luxury', fontsize=12)
    plt.title('Logistic Regression: House Luxury Classification vs. Area', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save a high-res copy for your repository
    plt.savefig('logistic_regression_curve.png', dpi=300, bbox_inches='tight')
    
    # Pop open the graph window!
    plt.show()
else:
    print("Error: Too few rows left after filtering.")
import joblib
joblib.dump(model, 'linear_regression_model.pkl')
print("Linear Regression model saved successfully!")