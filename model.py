import os
import glob
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Download and load
path = kagglehub.dataset_download("arvanshul/gurgaon-real-estate-99acres-com")
csv_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.read_csv(csv_files[0])

# --- DIAGNOSTIC: Print a few rows of area and price columns ---
print("Sample of area and price columns:")
print(df[['SUPER_SQFT', 'BUILTUP_SQFT', 'AREA', 'PRICE']].head(5))

# 1. Combine area columns using the correct dataset column names
df['Area'] = df['SUPER_SQFT'].fillna(df['BUILTUP_SQFT']).fillna(df['AREA'])
df['Amount(in rupees)'] = df['PRICE']

# 2. Clean area and price by forcing them to numeric (coercing errors to NaN)
df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
df['Amount(in rupees)'] = pd.to_numeric(df['Amount(in rupees)'], errors='coerce')

# 3. Drop rows where area or price is missing
df = df.dropna(subset=['Area', 'Amount(in rupees)'])

print(f"Rows remaining after cleaning: {len(df)}")

# 4. Now safe to filter numerically!
df = df[(df['Area'] > 50) & (df['Area'] < 15000) & (df['Amount(in rupees)'] > 100000) & (df['Amount(in rupees)'] < 50000000)]

print(f"Rows remaining after filtering: {len(df)}")

if len(df) > 10:
    X = df[['Area']]
    y = df['Amount(in rupees)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = r2_score(y_test, y_pred)
    print(f"\n--- Final Model Results ---")
    print(f"R-squared Score: {score:.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

    plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Prices')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Area (sqft)')
    plt.ylabel('Total Amount (in rupees)')
    plt.title('House Total Price vs Area')
    plt.legend()
    plt.show()
else:
    print("Error: Too few rows left after filtering. Check your dataset filters.")



# Sort test values so the regression line draws smoothly
sort_idx = X_test['Area'].argsort()
X_test_sorted = X_test.iloc[sort_idx]
y_pred_sorted = y_pred[sort_idx]

plt.figure(figsize=(10, 6)) # Makes the window a nice readable size
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Prices')
plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Line')

plt.xlabel('Area (sqft)')
plt.ylabel('Total Amount (in rupees)')
plt.title('House Total Price vs Area (Linear Regression)')
plt.legend()

# Save a high-res copy for your project README!
plt.savefig('house_price_regression.png', dpi=300, bbox_inches='tight')

plt.show()