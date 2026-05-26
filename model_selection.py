import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Train Test Split
from sklearn.model_selection import train_test_split

# Label Encoding
from sklearn.preprocessing import LabelEncoder

# Metrics
from sklearn.metrics import r2_score

# Models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------
df = pd.read_csv("data/football_players.csv")

# ---------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------
df['region'] = df['region'].fillna(
    df['region'].mode()[0]
)

# ---------------------------------------
# CLEAN fpl_sel
# ---------------------------------------
df['fpl_sel'] = df['fpl_sel'].astype(str)

df['fpl_sel'] = df['fpl_sel'].str.replace('%', '')

df['fpl_sel'] = pd.to_numeric(
    df['fpl_sel'],
    errors='coerce'
)

# ---------------------------------------
# ENCODE TEXT COLUMNS
# ---------------------------------------
le = LabelEncoder()

for column in df.columns:

    converted = pd.to_numeric(
        df[column],
        errors='coerce'
    )

    if converted.isnull().sum() > 0:

        df[column] = le.fit_transform(
            df[column].astype(str)
        )

# ---------------------------------------
# FEATURES AND TARGET
# ---------------------------------------
X = df.drop('market_value', axis=1)

y = df['market_value']

# ---------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# MODELS
# ---------------------------------------
models = {

    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(),

    "Lasso": Lasso(),

    "KNN": KNeighborsRegressor(),

    "SVR": SVR(),

    "Decision Tree": DecisionTreeRegressor(),

    "Random Forest": RandomForestRegressor(
        max_depth=15,
        min_samples_split=5,
        n_estimators=50,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=100,
        random_state=42
    )
}

# ---------------------------------------
# STORE RESULTS
# ---------------------------------------
results = {}

# ---------------------------------------
# TRAIN AND EVALUATE
# ---------------------------------------
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    results[name] = r2

# ---------------------------------------
# RESULTS TABLE
# ---------------------------------------
results_df = pd.DataFrame({

    'Model': results.keys(),

    'R2 Score': results.values()
})

print("\nMODEL COMPARISON\n")

print(results_df)

# ---------------------------------------
# BAR GRAPH
# ---------------------------------------
plt.figure(figsize=(12,6))

plt.bar(
    results.keys(),
    results.values()
)

plt.xticks(rotation=45)

plt.ylabel("R2 Score")

plt.title("Model Comparison")

plt.show()