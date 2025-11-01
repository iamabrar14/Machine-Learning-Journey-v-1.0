import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

# Generate columns
customer_id = np.arange(1, n+1)
age = np.random.randint(18, 70, size=n).astype(float)  # Cast to float for NaN assignment
gender = np.random.choice(['Male', 'Female', 'Other'], size=n).astype(object)
country = np.random.choice(['USA', 'Canada', 'UK', 'India', 'Australia', 'Germany'], size=n)
annual_income = np.round(np.random.normal(55000, 15000, size=n), 2)
membership = np.random.choice(['Silver', 'Gold', 'Platinum'], size=n).astype(object)
items_purchased = np.random.poisson(lam=5, size=n)
total_spent = np.round(items_purchased * np.random.normal(50, 15, size=n), 2)
last_purchase_days_ago = np.random.randint(1, 365, size=n)

# Add missing values (~10%)
for arr, val in zip(
    [age, gender, annual_income, membership],
    [np.nan, None, np.nan, None]
):
    mask = np.random.rand(n) < 0.1
    arr[mask] = val

df = pd.DataFrame({
    "CustomerID": customer_id,
    "Age": age,
    "Gender": gender,
    "Country": country,
    "AnnualIncome": annual_income,
    "Membership": membership,
    "ItemsPurchased": items_purchased,
    "TotalSpent": total_spent,
    "LastPurchaseDaysAgo": last_purchase_days_ago
})

df.to_csv('customer_purchases.csv', index=False)
print("Saved as customer_purchases.csv")
print(df.head())