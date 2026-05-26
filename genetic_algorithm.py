import pandas as pd
import numpy as np
import random

# Train Test Split
from sklearn.model_selection import train_test_split

# Label Encoding
from sklearn.preprocessing import LabelEncoder

# Metrics
from sklearn.metrics import mean_squared_error

# KNN
from sklearn.neighbors import KNeighborsRegressor

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
# GA PARAMETERS
# ---------------------------------------
population_size = 20

num_generations = 30

mutation_rate = 0.1

num_features = X.shape[1]

# ---------------------------------------
# CREATE INITIAL POPULATION
# ---------------------------------------
population = []

for _ in range(population_size):

    chromosome = np.random.rand(num_features)

    population.append(chromosome)

# ---------------------------------------
# FITNESS FUNCTION
# ---------------------------------------
def fitness(chromosome):

    # Apply weights
    weighted_X_train = X_train * chromosome

    weighted_X_test = X_test * chromosome

    # KNN Model
    model = KNeighborsRegressor(n_neighbors=5)

    model.fit(weighted_X_train, y_train)

    predictions = model.predict(weighted_X_test)

    mse = mean_squared_error(y_test, predictions)

    # Lower error = better fitness
    return 1 / (mse + 1)

# ---------------------------------------
# SELECTION
# ---------------------------------------
def selection(population, fitness_scores):

    selected = np.random.choice(
        len(population),
        size=2,
        p=fitness_scores / np.sum(fitness_scores)
    )

    return population[selected[0]], population[selected[1]]

# ---------------------------------------
# CROSSOVER
# ---------------------------------------
def crossover(parent1, parent2):

    crossover_point = random.randint(
        1,
        num_features - 1
    )

    child = np.concatenate([
        parent1[:crossover_point],
        parent2[crossover_point:]
    ])

    return child

# ---------------------------------------
# MUTATION
# ---------------------------------------
def mutation(chromosome):

    for i in range(num_features):

        if random.random() < mutation_rate:

            chromosome[i] = random.random()

    return chromosome

# ---------------------------------------
# GENETIC ALGORITHM LOOP
# ---------------------------------------
best_score = 0

best_chromosome = None

for generation in range(num_generations):

    fitness_scores = np.array([

        fitness(chromosome)

        for chromosome in population
    ])

    # Best solution
    best_index = np.argmax(fitness_scores)

    if fitness_scores[best_index] > best_score:

        best_score = fitness_scores[best_index]

        best_chromosome = population[best_index]

    # Create next generation
    new_population = []

    for _ in range(population_size):

        parent1, parent2 = selection(
            population,
            fitness_scores
        )

        child = crossover(parent1, parent2)

        child = mutation(child)

        new_population.append(child)

    # -----------------------------------
    # DIVERSITY MAINTENANCE
    # -----------------------------------
    # Add random chromosomes
    for _ in range(2):

        random_chromosome = np.random.rand(
            num_features
        )

        new_population.append(random_chromosome)

    population = new_population[:population_size]

    print(f"Generation {generation+1} Completed")

# ---------------------------------------
# FINAL RESULTS
# ---------------------------------------
print("\n")
print("="*60)

print("BEST FEATURE WEIGHTS")

print("="*60)

print(best_chromosome)

print("\nBest Fitness Score:")

print(best_score)