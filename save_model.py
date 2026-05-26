import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

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
# BEST RANDOM FOREST MODEL
# ---------------------------------------
model = RandomForestRegressor(
    max_depth=15,
    min_samples_split=5,
    n_estimators=50,
    random_state=42
)

# TRAIN MODEL
model.fit(X_train, y_train)

# SAVE MODEL
joblib.dump(model, "football_model.pkl")

print("Model Saved Successfully")