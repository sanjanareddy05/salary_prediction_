# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_csv(r'C:\Users\sanja\Downloads\salary_prediction\Salary_Data.csv')

# =====================
# Data Cleaning
# =====================
# Fix Education Level values
df['Education Level'] = df['Education Level'].replace({
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    "phD": "PhD"
})

# Drop nulls
df.dropna(inplace=True)

# Combine rare job titles into 'Others'
job_counts = df['Job Title'].value_counts()
rare_jobs = job_counts[job_counts < 25].index
df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in rare_jobs else x)

# Label encode Gender
df['Gender'] = LabelEncoder().fit_transform(df['Gender'].astype(str))

# Map Education Level
education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df['Education Level'] = df['Education Level'].map(education_map)

# =====================
# Feature Engineering
# =====================
# One-hot encode Job Title
df = pd.get_dummies(df, columns=['Job Title'], drop_first=True)

# Split features & target
X = df.drop('Salary', axis=1)
y = df['Salary']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# =====================
# Model Training & Evaluation
# =====================
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=2, random_state=0),
    'Random Forest': RandomForestRegressor(n_estimators=20, random_state=0)
}

results = []

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    results.append({
        'Model': name,
        'R² Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    })

results_df = pd.DataFrame(results).sort_values(by='R² Score', ascending=False)
print(results_df)

# =====================
# Feature Importance (Random Forest)
# =====================
best_model = models['Random Forest']
importances = best_model.feature_importances_
feature_names = X.columns

# Plot top 10
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()
import joblib
joblib.dump(best_model, 'best_salary_model.pkl')

