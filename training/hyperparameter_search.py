import os
import pickle
import csv
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, "dataset", "data.pickle")

with open(DATA_FILE, "rb") as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# IMPORTANTE:
# A grade original possui 995.328 combinações.
# Com cv=10, isso pode gerar milhões de treinamentos.
# Esta versão mantém a ideia, mas começa com uma grade controlada.
param_grid = {
    "n_estimators": [100, 250, 750],
    "max_depth": [None, 10, 50],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True],
    "criterion": ["gini", "entropy"],
}

param_combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["min_samples_split"],
    param_grid["min_samples_leaf"],
    param_grid["max_features"],
    param_grid["bootstrap"],
    param_grid["criterion"],
))

print(f"Testando {len(param_combinations)} combinações.")

results = []

for i, params in enumerate(param_combinations, start=1):
    (
        n_estimators, max_depth, min_samples_split,
        min_samples_leaf, max_features, bootstrap, criterion
    ) = params

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        random_state=42,
        n_jobs=-1
    )

    scores = cross_val_score(
        model, x_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )

    mean_score = scores.mean()
    std_score = scores.std()

    results.append({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "criterion": criterion,
        "mean_accuracy": mean_score,
        "std_accuracy": std_score
    })

    print(f"{i}/{len(param_combinations)} -> {mean_score:.4f}")

results.sort(key=lambda x: (-x["mean_accuracy"], x["std_accuracy"]))

csv_file = os.path.join(ROOT, "training", "hyperparameter_results.csv")

with open(csv_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Resultados salvos: {csv_file}")
