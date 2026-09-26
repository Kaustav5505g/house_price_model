import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('house_prices.csv')

def clean_area(val):
    if pd.isna(val):
        return None
    val_str = str(val).replace('sqft', '').replace('sq ft', '').strip()
    try:
        return float(val_str)
    except:
        return None

for col in ['Super Area', 'Carpet Area', 'Plot Area']:
    if col in df.columns:
        df[col] = df[col].apply(clean_area)

df['Area'] = df['Super Area'].fillna(df['Carpet Area']).fillna(df['Plot Area'])

def clean_amount(val):
    if pd.isna(val):
        return None
    val_str = str(val).replace(',', '').replace('₹', '').strip()
    multiplier = 1.0
    if 'cr' in val_str.lower():
        multiplier = 10000000
        val_str = val_str.lower().replace('cr', '').strip()
    elif 'lac' in val_str.lower() or 'lakh' in val_str.lower():
        multiplier = 100000
        val_str = val_str.lower().replace('lac', '').replace('lakh', '').strip()
    
    try:
        return float(val_str) * multiplier
    except:
        return None

df['Amount(in rupees)'] = df['Amount(in rupees)'].apply(clean_amount)

df = df.dropna(subset=['Area', 'Amount(in rupees)'])

print(f"Rows remaining after cleaning: {len(df)}")

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