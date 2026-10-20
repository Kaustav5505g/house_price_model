import os
import glob
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Automatically download dataset via kagglehub
path = kagglehub.dataset_download("arvanshul/gurgaon-real-estate-99acres-com")
csv_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.read_csv(csv_files[0])

print(f"Total rows loaded: {len(df)}")

# 2. Combine area columns using dataset column names
df['Area'] = pd.to_numeric(df['SUPER_SQFT'], errors='coerce') \
    .fillna(pd.to_numeric(df['BUILTUP_SQFT'], errors='coerce')) \
    .fillna(pd.to_numeric(df['AREA'], errors='coerce'))

# 3. Clean price (PRICE column in this dataset is in Lakhs)
df['Amount(in lakhs)'] = pd.to_numeric(df['PRICE'], errors='coerce')

# Drop missing values
df = df.dropna(subset=['Area', 'Amount(in lakhs)'])
print(f"Rows remaining after cleaning: {len(df)}")

# 4. Filter realistic ranges (Area between 100-15000 sqft, Price between 10 Lakhs and 500 Lakhs / 5 Crores)
df = df[(df['Area'] > 100) & (df['Area'] < 15000) & (df['Amount(in lakhs)'] > 10) & (df['Amount(in lakhs)'] < 500)]
print(f"Rows remaining after filtering: {len(df)}")

if len(df) > 10:
    X = df[['Area']]
    # Convert Lakhs to raw rupees for clean plotting and metrics
    y = df['Amount(in lakhs)'] * 100000  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = r2_score(y_test, y_pred)
    print(f"\n--- Final Model Results ---")
    print(f"R-squared Score: {score:.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

    # Sort test values so the regression line draws smoothly without zigzagging
    sort_idx = X_test['Area'].argsort()
    X_test_sorted = X_test.iloc[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Prices')
    plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Area (sqft)')
    plt.ylabel('Total Amount (in rupees)')
    plt.title('House Total Price vs Area (Linear Regression)')
    plt.legend()
    
    # Save a high-res copy for your GitHub README
    plt.savefig('house_price_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("Error: Too few rows left after filtering. Check your dataset filters.")