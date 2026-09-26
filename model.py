import pandas as pd 
df = pd.read_csv('house_prices.csv')
print("Column Names:")
print(df.columns)
print("\nFirst 3 rows:")
print(df.head(3))