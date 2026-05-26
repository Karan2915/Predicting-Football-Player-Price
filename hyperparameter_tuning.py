import pandas as pd
import numpy as np

# Train Test Split
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

# Encoding
from sklearn.preprocessing import LabelEncoder

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Models
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
# CLEAN fpl_sel COLUMN
# ---------------------------------------
df['fpl_sel'] = df['fpl_sel'].astype(str)

df['fpl_sel'] = df['fpl_sel'].str.replace('%', '')

df['fpl_sel'] = pd.to_numeric(
    df['fpl_sel'],
    errors='coerce'
)

# ---------------------------------------
# ENCODE NON NUMERIC COLUMNS
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

# =====================================================
# RANDOM FOREST TUNING
# =====================================================

rf = RandomForestRegressor()

rf_params = {

    'n_estimators': [50, 100],

    'max_depth': [5, 10, 15],

    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(
    rf,
    rf_params,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

# Best RF Model
best_rf = rf_grid.best_estimator_

# Prediction
rf_predictions = best_rf.predict(X_test)

# Metrics
rf_r2 = r2_score(y_test, rf_predictions)

print("\n")
print("="*60)

print("BEST RANDOM FOREST")

print("="*60)

print("Best Parameters:")

print(rf_grid.best_params_)

print(f"R2 Score : {rf_r2:.2f}")

# =====================================================
# GRADIENT BOOSTING TUNING
# =====================================================

gb = GradientBoostingRegressor()

gb_params = {

    'n_estimators': [50, 100],

    'learning_rate': [0.01, 0.1],

    'max_depth': [3, 5]
}

gb_grid = GridSearchCV(
    gb,
    gb_params,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)

# Best GB Model
best_gb = gb_grid.best_estimator_

# Prediction
gb_predictions = best_gb.predict(X_test)

# Metrics
gb_r2 = r2_score(y_test, gb_predictions)

print("\n")
print("="*60)

print("BEST GRADIENT BOOSTING")

print("="*60)

print("Best Parameters:")

print(gb_grid.best_params_)

print(f"R2 Score : {gb_r2:.2f}")