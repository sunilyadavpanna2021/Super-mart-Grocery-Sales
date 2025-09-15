
# Supermart Grocery Sales - Retail Analytics Dataset Project

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the Dataset
data = pd.read_csv('supermart_grocery_sales.csv')  # Update path if necessary

print("First 5 rows of dataset:")
print(data.head())

# Step 3: Data Preprocessing

# 3.1 Check for missing values and handle them
print("\nMissing values:")
print(data.isnull().sum())

data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# 3.2 Convert 'Order Date' to datetime and extract day, month, year
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Order Day'] = data['Order Date'].dt.day
data['Order Month'] = data['Order Date'].dt.month
data['Order Year'] = data['Order Date'].dt.year

# Also extract month name and numeric month
data['month_no'] = data['Order Date'].dt.month
data['Month'] = data['Order Date'].dt.strftime('%B')
data['year'] = data['Order Date'].dt.year

# 3.3 Label Encoding for categorical variables
le = LabelEncoder()

categorical_columns = ['Category', 'Sub Category', 'City', 'Region', 'State', 'Month']
for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

print("\nData after encoding:")
print(data.head())

# Step 4: Exploratory Data Analysis (EDA)

# 4.1 Distribution of Sales by Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Sales', data=data, palette='Set2')
plt.title('Sales Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()

# 4.2 Sales Trends Over Time
plt.figure(figsize=(12, 6))
data.groupby('Order Date')['Sales'].sum().plot()
plt.title('Total Sales Over Time')
plt.xlabel('Order Date')
plt.ylabel('Sales')
plt.show()

# 4.3 Correlation Heatmap
plt.figure(figsize=(12, 6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 5: Feature Selection and Model Preparation

# Features and target
features = data.drop(columns=['Order ID', 'Customer Name', 'Order Date', 'Sales'])
target = data['Sales']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Model Building

model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Model Evaluation

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Step 8: Visualize Results

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

# Sales by Month
monthly_sales = data.groupby('Month')['Sales'].sum().reset_index()
monthly_sales_sorted = monthly_sales.sort_values(by='Month')

plt.figure(figsize=(10, 6))
plt.plot(monthly_sales_sorted['Month'], monthly_sales_sorted['Sales'], marker='o')
plt.title('Sales by Month')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Yearly Sales
yearly_sales = data.groupby('year')['Sales'].sum()
plt.figure(figsize=(8, 6))
plt.pie(yearly_sales, labels=yearly_sales.index, autopct='%1.1f%%')
plt.title('Sales by Year')
plt.show()

# Top 5 Cities by Sales
city_sales = data.groupby('City')['Sales'].sum()
top_cities = city_sales.sort_values(ascending=False).head(5)

plt.figure(figsize=(10, 6))
plt.bar(top_cities.index, top_cities.values)
plt.title('Top 5 Cities by Sales')
plt.xlabel('City')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.show()

# Sales by Category
category_sales = data.groupby('Category')['Sales'].sum()
plt.figure(figsize=(10, 6))
category_sales.plot(kind='bar')
plt.title('Sales by Category')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()

# Step 9: Conclusion

print("\nConclusion:")
print("✔ The linear regression model provides reasonable predictions for supermarket sales.")
print(f"✔ Model R-squared value: {r2:.2f}, indicating how well the model explains the variability in sales.")
print("✔ Visualization of trends, correlation, and distribution helps in understanding the data patterns.")
print("✔ Further improvements can be made by experimenting with advanced models and feature engineering.")

# End of project
