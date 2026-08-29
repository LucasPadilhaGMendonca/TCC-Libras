import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report

from grouping import min_groups_per_class

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, "dataset", "data.pickle")
MODEL_FILE = os.path.join(ROOT, "training", "model.p")

if not os.path.exists(DATA_FILE):
    raise SystemExit(
        "dataset/data.pickle não encontrado. Execute create_dataset.py primeiro."
    )

with open(DATA_FILE, "rb") as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict["data"], dtype=np.float32)
labels = np.asarray(data_dict["labels"])
# "groups" identifica o vídeo/intérprete de origem de cada frame (ver
# create_dataset.py). Pickles antigos sem essa chave caem no fallback abaixo.
groups = np.asarray(data_dict.get("groups", labels))

if min_groups_per_class(labels, groups) < 2:
    print(
        "[AVISO] Pelo menos uma classe tem frames de um único vídeo/intérprete. "
        "Não é possível fazer um split por grupo que deixe cada classe "
        "representada em treino e teste sem vazamento. Usando split aleatório "
        "por frame (ver README, seção de dataset: mais vídeos por classe "
        "deixam essa métrica confiável)."
    )
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        shuffle=True,
        stratify=labels,
        random_state=42
    )
else:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(data, labels, groups))
    x_train, x_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    print(
        "Split por grupo (StratifiedGroupKFold): nenhum vídeo/intérprete "
        "aparece simultaneamente em treino e teste."
    )

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    criterion="gini",
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)

print(f"Acurácia: {accuracy * 100:.2f}%")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_predict, zero_division=0))

with open(MODEL_FILE, "wb") as f:
    pickle.dump({"model": model}, f)

print(f"Modelo salvo: {MODEL_FILE}")
