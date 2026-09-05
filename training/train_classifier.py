import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_FILE = "dataset/data.pickle"
MODEL_OUTPUT = "training/model.p"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} não encontrado. Execute create_dataset.py primeiro.")

with open(DATA_FILE, "rb") as f:
    dataset = pickle.load(f)

X = np.asarray(dataset["data"], dtype=np.float32)
y = np.asarray(dataset["labels"])

print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features por vetor.")

# Separação Treino/Teste estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=25,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

print("Treinando classificador Random Forest...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAcurácia no conjunto de teste: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump({"model": model, "labels": sorted(list(set(y)))}, f)

print(f"Modelo salvo: {os.path.abspath(MODEL_OUTPUT)}")