# Machine Learning Practice Datasets

This directory contains 6 synthetic CSV datasets, each with 5000 rows, designed for machine learning practice. All datasets contain realistic data with proper value ranges and correlations.

## Dataset Descriptions

### 1. house_prices.csv (Regression)
**Use case:** Predicting house prices based on property features  
**Columns:**
- `rooms`: Number of rooms (1-7)
- `area`: Floor area in square feet (300-3000)
- `age`: Age of house in years (0-50)
- `price`: House price in USD (50,000-1,000,000)

**Python loading example:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('datasets/house_prices.csv')

# Prepare features (X) and target (y)
X = df[['rooms', 'area', 'age']]
y = df['price']

# Split for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. student_pass.csv (Binary Classification)
**Use case:** Predicting student pass/fail based on study habits  
**Columns:**
- `hours_studied`: Hours studied per week (0-100)
- `attendance`: Attendance percentage (30-100)
- `assignments_completed`: Number of assignments completed (0-20)
- `pass`: Pass status (0=Fail, 1=Pass)

**Python loading example:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('datasets/student_pass.csv')

# Prepare features (X) and target (y)
X = df[['hours_studied', 'attendance', 'assignments_completed']]
y = df['pass']

# Split for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. fruit_knn.csv (KNN Classification)
**Use case:** Classifying fruits using K-Nearest Neighbors  
**Columns:**
- `size_cm`: Fruit size in centimeters (3-25)
- `shade`: Color shade intensity (1-10)
- `fruit`: Fruit type ('apple', 'orange', 'banana')

**Python loading example:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('datasets/fruit_knn.csv')

# Prepare features (X) and target (y)
X = df[['size_cm', 'shade']]
y = df['fruit']

# Split for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. loan_approval.csv (Binary Classification)
**Use case:** Predicting loan approval decisions  
**Columns:**
- `income`: Annual income in USD (20,000-300,000)
- `credit_score`: Credit score (300-850)
- `age`: Age in years (18-80)
- `loan_amount`: Requested loan amount in USD (10,000-1,000,000)
- `approved`: Approval status (0=Rejected, 1=Approved)

**Python loading example:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('datasets/loan_approval.csv')

# Prepare features (X) and target (y)
X = df[['income', 'credit_score', 'age', 'loan_amount']]
y = df['approved']

# Split for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. wine_quality.csv (Multiclass Classification/Regression)
**Use case:** Predicting wine quality ratings or classification  
**Columns:**
- `fixed_acidity`: Fixed acidity level (4.6-15.9)
- `volatile_acidity`: Volatile acidity level (0.12-1.58)
- `citric_acid`: Citric acid content (0-1)
- `residual_sugar`: Residual sugar content (0.9-15.5)
- `chlorides`: Chloride content (0.012-0.611)
- `alcohol`: Alcohol percentage (8.4-14.9)
- `quality`: Wine quality score (3-9)

**Python loading example:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('datasets/wine_quality.csv')

# Prepare features (X) and target (y)
X = df[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'alcohol']]
y = df['quality']

# Split for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 6. mall_customers.csv (Clustering)
**Use case:** Customer segmentation using clustering algorithms  
**Columns:**
- `customer_id`: Unique customer identifier (1-5000)
- `gender`: Customer gender ('Male', 'Female')
- `age`: Customer age (18-70)
- `annual_income`: Annual income in USD (15,000-140,000)
- `spending_score`: Spending score (1-100)

**Python loading example:**
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('datasets/mall_customers.csv')

# For clustering, typically exclude ID and encode categorical variables
le = LabelEncoder()
df_cluster = df.copy()
df_cluster['gender_encoded'] = le.fit_transform(df['gender'])

# Prepare features for clustering (excluding customer_id)
X = df_cluster[['gender_encoded', 'age', 'annual_income', 'spending_score']]

# Alternative: Use only numerical features
X_numerical = df[['age', 'annual_income', 'spending_score']]
```

## General Tips

1. **Data Preprocessing**: Remember to scale/normalize features when using algorithms sensitive to feature magnitude (SVM, KNN, Neural Networks).

2. **Categorical Encoding**: For datasets with categorical variables (like `fruit` or `gender`), use appropriate encoding methods:
   - Label Encoding for ordinal data
   - One-Hot Encoding for nominal data

3. **Train-Test Split**: Always split your data into training and testing sets to evaluate model performance.

4. **Feature Engineering**: Consider creating additional features from existing ones to improve model performance.

5. **Cross-Validation**: Use cross-validation for more robust model evaluation, especially with smaller datasets.

## Example: Complete ML Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load house prices dataset
df = pd.read_csv('datasets/house_prices.csv')

# Prepare data
X = df[['rooms', 'area', 'age']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")
```

Each dataset is designed to provide realistic practice scenarios for different types of machine learning problems. Happy learning!