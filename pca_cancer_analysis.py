import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_breast_cancer()      # Uses load_breast_cancer() from sklearn.datasets to import the dataset.
X = data.data
y = data.target

# Standardize the dataset
scaler = StandardScaler()          # Uses StandardScaler() to scale the data so that all features have equal importance.
X_scaled = scaler.fit_transform(X)

# Apply PCA with 2 components
pca = PCA(n_components=2)                         # Uses PCA(n_components=2) to reduce the 30 original features to just 2 principal components.
X_pca = pca.fit_transform(X_scaled)              # The transformed data is stored in a new DataFrame.

# Convert to DataFrame for visualization
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Target'] = y

# Scatter plot of PCA components
plt.figure(figsize=(8,6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Target'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Principal Component 1')                     #Uses seaborn.scatterplot() to create a scatter plot of the two principal components.
plt.ylabel('Principal Component 2')
plt.title('PCA of Breast Cancer Dataset')
plt.colorbar(label='Target')
plt.show()

# Bonus: Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)    # Splits the PCA-transformed dataset into training and test sets.
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)                  #Trains a logistic regression model to predict tumor diagnosis.

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy (using PCA features): {accuracy:.2f}')     # Prints the model’s accuracy
