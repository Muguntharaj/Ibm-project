import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
#leads to proceed
# Load the data
data = pd.read_csv('/content/daily-website-visitors.csv')

# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])
data['day_num'] = (data['date'] - data['date'].min()).dt.days
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month

# Check for missing values
print(data.isnull().sum())

# Descriptive statistics
print(data.describe())

# Visualize data distributions
sns.pairplot(data)
plt.show()

# Linear Regression Forecasting

# Splitting the data
X = data[['day_num']]
y = data['page_views']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualizing Actual vs Predicted
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Page Views")
plt.show()

# Decision Tree Classifier

# Assuming the 'converted' column exists
X = data[['session_duration', 'pages_per_session']]
y = data['converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Visualizing the tree
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=['session_duration', 'pages_per_session'], class_names=['not_converted', 'converted'])
plt.show()
