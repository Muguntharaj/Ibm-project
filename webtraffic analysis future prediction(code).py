import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/content/daily-website-visitors.csv')

# Time series prediction using Linear Regression for page_views
data['date'] = pd.to_datetime(data['date'])
data['day_num'] = (data['date'] - data['date'].min()).dt.days  # convert dates to day numbers
model = LinearRegression()
model.fit(data[['day_num']], data['page_views'])

# Predict next 7 days
next_week = pd.DataFrame({'day_num': np.arange(data['day_num'].max()+1, data['day_num'].max()+8)})
predictions = model.predict(next_week)

plt.plot(data['date'], data['page_views'], label='Actual Traffic')
plt.plot(pd.date_range(data['date'].iloc[-1], periods=8)[1:], predictions, linestyle='--', label='Predicted Traffic')
plt.xlabel('Date')
plt.ylabel('Page Views')
plt.title('Traffic Forecast using Linear Regression')
plt.legend()
plt.show()

# Predict user behavior with DecisionTreeClassifier
# Assuming we have a column "converted" indicating if the user converted (1) or not (0)
X = data[['session_duration', 'pages_per_session']]
y = data['converted']
clf = DecisionTreeClassifier(max_depth=3)  # Limiting depth for visualization
clf.fit(X, y)

plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=['session_duration', 'pages_per_session'], class_names=['not_converted', 'converted'])
plt.title('User Behavior using Decision Tree')
plt.show()
