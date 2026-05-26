import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/football_players.csv")

# Fill missing value
df['region'].fillna(df['region'].mode()[0], inplace=True)

# -------------------------------
# 1. Dataset Overview
# -------------------------------
print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES")
print(df.isnull().sum())

# -------------------------------
# 2. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 3. Distribution of Market Value
# -------------------------------
plt.figure(figsize=(8,5))

sns.histplot(df['market_value'], kde=True)

plt.title("Distribution of Market Value")
plt.xlabel("Market Value")
plt.ylabel("Count")

plt.show()

# -------------------------------
# 4. Age vs Market Value
# -------------------------------
plt.figure(figsize=(8,5))

sns.scatterplot(
    x='age',
    y='market_value',
    data=df
)

plt.title("Age vs Market Value")

plt.show()

# -------------------------------
# 5. Position Category Count
# -------------------------------
plt.figure(figsize=(8,5))

sns.countplot(
    x='position_cat',
    data=df
)

plt.title("Position Category Count")

plt.show()

# -------------------------------
# 6. Boxplot for Big Clubs
# -------------------------------
plt.figure(figsize=(8,5))

sns.boxplot(
    x='big_club',
    y='market_value',
    data=df
)

plt.title("Big Club vs Market Value")

plt.show()