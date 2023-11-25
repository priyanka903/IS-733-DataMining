#!/usr/bin/env python
# coding: utf-8

# # Homework_3

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


# displaying the df_sp500_close dataset
df_sp500_close=pd.read_csv('SP500_close_price_no_missing.csv')
df_sp500_close


# In[3]:


# Column names displayed
df_sp500_close.columns


# In[4]:


df_sp500_close.transpose()


# In[5]:


df_sp500_close= df_sp500_close.convert_dtypes()
df_sp500_close['date']= pd.to_datetime(df_sp500_close['date'])
df_sp500_close.reset_index(inplace=True)
df_sp500_close.dtypes


# In[6]:


# displaying the df_sp500_ticket dataset
df_sp500_ticket = pd.read_csv('SP500_ticker.csv', encoding='latin1')
df_sp500_ticket


# In[7]:


# Column names displayed
df_sp500_ticket.columns


# In[8]:


df_sp500_ticket.transpose()


# # Problem 1

# #### Fit a PCA model to log returns  (log return = log( Price [t+1]/Price [t]) derived from stock price data and complete the following tasks

# In[9]:


# Calculate log returns with a small constant added to avoid division by zero
log_returns = np.log((df_sp500_close.set_index('date') + 1e-8) / (df_sp500_close.set_index('date').shift(1) + 1e-8))

# Drop the first row with NaN values
log_returns = log_returns.dropna()


# In[10]:


# Standardize the data
scaler = StandardScaler()
log_returns_standardized = scaler.fit_transform(log_returns)

# Apply PCA
pca = PCA()
pca.fit(log_returns_standardized)


# #### Plot a scree plot which shows the distribution of variance contained in subsequent principal components sorted by their eigenvalues.
# 

# In[11]:


# Plot a scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues (Variance)')
plt.show()


# #### Create a second plot showing cumulative variance retained if top N components are kept after dimensionality reduction

# In[12]:


# Create a plot showing cumulative variance retained
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.title('Cumulative Variance Retained')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Ratio')
plt.grid(True)
plt.show()


# #### No of principal components must be retained in order to capture at least 80% of the total variance in data

# In[13]:


# Determine the number of principal components for 80% variance retention
n_components_80_percent = np.argmax(cumulative_variance_ratio >= 0.8) + 1
print(f"Number of components to retain at least 80% of the total variance: {n_components_80_percent}")


# ## Analysis of principal components and weights

# #### Compute and plot the time series of the 1st principal component and observe temporal patterns. Identify the date with the lowest value for this component and conduct a quick research on the Internet to see if you can identify event(s) that might explain the observed behavior

# In[14]:


# Get the principal components
principal_components = pca.transform(log_returns_standardized)

# Extract the 1st principal component
pc1 = principal_components[:, 0]

dates = df_sp500_close['date'].iloc[:-1]

plt.figure(figsize=(10, 6))
plt.plot(dates, pc1, label='1st Principal Component')
plt.title('Time Series of 1st Principal Component')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Rest of the code remains the same


# #### Extract the weights from the PCA model for 1st and 2nd principal components

# In[15]:


# Extract weights for the 1st and 2nd principal components
weights_pc1 = pca.components_[0, :]
weights_pc2 = pca.components_[1, :]
weights_pc1


# In[16]:


weights_pc2


# #### Create a plot to show weights of the 1st principal component grouped by the industry sector (for example, you may draw a bar plot of mean weight per sector). Observe the distribution of weights (magnitudes, signs). Based on your observation, what kind of information do you think the 1st principal component might have captured

# In[17]:


# Ensure the lengths match by considering only the relevant rows in df_sp500_ticket
df_weights_pc1 = pd.DataFrame({'Sector': df_sp500_ticket['sector'].iloc[:len(weights_pc1)], 'Weight_PC1': weights_pc1})

# Plot weights of the 1st principal component grouped by industry sector using matplotlib
plt.figure(figsize=(12, 6))
plt.bar(df_weights_pc1['Sector'], df_weights_pc1['Weight_PC1'])
plt.title('Weights of 1st Principal Component by Industry Sector')
plt.xlabel('Industry Sector')
plt.ylabel('Mean Weight')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# #### Make a similar plot for the 2nd principal component.  What kind of information do you think does this component reveal?

# In[18]:


# Extract weights for the 2nd principal component
weights_pc2 = pca.components_[1, :]

# Ensure the lengths match by considering only the relevant rows in df_sp500_ticket
df_weights_pc2 = pd.DataFrame({'Sector': df_sp500_ticket['sector'].iloc[:len(weights_pc2)], 'Weight_PC2': weights_pc2})

# Plot weights of the 2nd principal component grouped by industry sector using matplotlib
plt.figure(figsize=(12, 6))
bars = plt.bar(df_weights_pc2['Sector'], df_weights_pc2['Weight_PC2'])

# Add a horizontal line at zero
plt.axhline(0, color='white', linestyle='-', linewidth=2, label='Zero Line')

plt.title('Weights of 2nd Principal Component by Industry Sector')
plt.xlabel('Industry Sector')
plt.ylabel('Mean Weight')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.legend()  # Show legend with the zero line
plt.show()


# In[19]:


# Calculate explained variance for the 1st and 2nd principal components
explained_variance_pc1 = pca.explained_variance_ratio_[0]
explained_variance_pc2 = pca.explained_variance_ratio_[1]

print(f'Explained Variance - 1st Principal Component: {explained_variance_pc1:.2%}')
print(f'Explained Variance - 2nd Principal Component: {explained_variance_pc2:.2%}')


# #### Suppose we wanted to construct a new stock index using one principal component to track the overall market tendencies. Which of the two components would you prefer to use for this purpose, the 1st or the 2nd? Why?

# > When the first principal component explains a significantly larger portion of the total variance compared to subsequent components, it indicates that it captures a substantial amount of the variability present in the original data. This makes it a valuable tool for constructing a new stock index that effectively tracks overall market tendencies. By condensing a substantial amount of information from the original dataset into a single variable, the first principal component provides a concise representation of the primary direction of market movements

# ### Problem 2 

# In[20]:


from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df_bmi= pd.read_csv('BMI.csv')
df_bmi


# In[21]:


# Split the data into features (X) and target variable (y)
X = df_bmi.drop('fatpctg', axis=1)
y = df_bmi['fatpctg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# #### Wrapper method

# In[22]:


# Backward stepwise regression
model = LinearRegression()
backward_selector = RFE(model, n_features_to_select=1)
backward_selector = backward_selector.fit(X_train, y_train)
backward_features = X.columns[backward_selector.support_]
print("Wrapper Method - Backward Selection Features:", backward_features)


# In[23]:


# Forward stepwise regression
forward_selector = RFE(model, n_features_to_select=1, step=1)
forward_selector = forward_selector.fit(X_train, y_train)
forward_features = X.columns[forward_selector.support_]
print("Wrapper Method - Forward Selection Features:", forward_features)


# #### Filter method

# In[24]:


# Filter method: Correlation statistics
correlation_ranking = X.corrwith(y).abs().sort_values(ascending=False)
print("Filter Method - Correlation Ranking:", correlation_ranking)


# #### Embedded method

# In[25]:


# Embedded method: Lasso regression
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)
lasso_features = X.columns[lasso_model.coef_ != 0]
print("Embedded Method - Lasso Regression Features:", lasso_features)


# In[26]:


# Embedded method: Random forest feature importance
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_features = X.columns[rf_model.feature_importances_.argsort()[::-1]]
print("Embedded Method - Random Forest Feature Importance:", rf_features)


# ### Write a paragraph to summarize your findings from the above experiments.

# > Feature selection experiments on the BMI dataset revealed the critical factors influencing fat percentage prediction. The wrapper method identified 'Wrist' as the most significant predictor, while the filter method highlighted 'Abdomen' and 'Chest' for their strong correlations with fat percentage. Embedded methods, including Lasso regression and random forest feature importance, emphasized the importance of 'Abdomen,' 'Wrist,' 'Height,' and 'Weight.' Combining insights from all three methods provides a comprehensive understanding of the relevant features for developing robust fat percentage prediction models
