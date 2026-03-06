from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    f1_score
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/ekaterinaselihovkina/Documents/GitHub/ML2_project/WineQuality.csv')
cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'quality']

sns.set_theme(style="whitegrid")
plt.figure(figsize=(20, 15))
plt.suptitle('Distr', fontsize=20, y=1.02)

for i, col in enumerate(cols):
    plt.subplot(4, 3, i + 1)
    sns.histplot(df[col], kde=True, color='skyblue', bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()

#outliers
plt.figure(figsize=(20, 15))

for i, col in enumerate(cols):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(y=df[col], color='lightcoral')
    plt.title(f'Outliers in {col}')
    plt.ylabel('')

plt.tight_layout()
plt.show()


corr = df["alcohol"].corr(df["quality"], method="pearson")
corr_matrix = df.select_dtypes(include='number').corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


X = df.iloc[:, :-1] 
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")


rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")