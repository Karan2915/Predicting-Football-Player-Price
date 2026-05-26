import pandas as pd
import numpy as np

# Train Test Split
from sklearn.model_selection import train_test_split

# Label Encoding
from sklearn.preprocessing import LabelEncoder

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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
# CLEAN fpl_sel COLUMN
# ---------------------------------------
df['fpl_sel'] = df['fpl_sel'].astype(str)

df['fpl_sel'] = df['fpl_sel'].str.replace('%', '')

df['fpl_sel'] = pd.to_numeric(
    df['fpl_sel'],
    errors='coerce'
)

# ---------------------------------------
# ENCODE EVERY NON-NUMERIC COLUMN
# ---------------------------------------
le = LabelEncoder()

for column in df.columns:

    # Try converting to numeric
    converted = pd.to_numeric(
        df[column],
        errors='coerce'
    )

    # If conversion fails -> encode text
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

    "Ridge Regression": Ridge(),

    "Lasso Regression": Lasso(),

    "KNN Regression": KNeighborsRegressor(),

    "SVR": SVR(),

    "Decision Tree": DecisionTreeRegressor(),

    "Random Forest": RandomForestRegressor(),

    "Gradient Boosting": GradientBoostingRegressor()
}

# ---------------------------------------
# TRAIN AND EVALUATE
# ---------------------------------------
for name, model in models.items():

    print("\n")
    print("=" * 60)

    print(name)

    print("=" * 60)

    # TRAIN
    model.fit(X_train, y_train)

    # PREDICT
    predictions = model.predict(X_test)

    # METRICS
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    # RESULTS
    print(f"MAE       : {mae:.2f}")

    print(f"MSE       : {mse:.2f}")

    print(f"RMSE      : {rmse:.2f}")

    print(f"R2 Score  : {r2:.2f}")