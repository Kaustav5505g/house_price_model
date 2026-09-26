import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('house_prices.csv')

df['Super Area'] = df['Super Area'].astype(str).str.replace('sqft', '', case=False).str.strip()
df['Super Area'] = pd.to_numeric(df['Super Area'], errors='coerce')

df['Price (in rupees)'] = pd.to_numeric(df['Price (in rupees)'], errors='coerce')

df = df.dropna(subset=['Super Area', 'Price (in rupees)'])

X = df[['Super Area']]
y = df['Price (in rupees)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Results ---")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Super Area')
plt.ylabel('Price (in rupees)')
plt.title('House Price vs Super Area (Linear Regression)')
plt.legend()
plt.show()