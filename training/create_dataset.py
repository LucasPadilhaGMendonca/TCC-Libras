import os
import pickle
import numpy as np

def build_features(hand_landmarks):
    """
    Normaliza os marcos de uma mão em relação ao bounding box mínimo (minX, minY).
    Retorna 42 floats para uma mão ou vetor combinado.
    """
    if not hand_landmarks:
        return None

    # Suporte tanto a objetos landmark com atributos .x/.y quanto dicionários/listas
    xs = [p.x if hasattr(p, 'x') else p[0] for p in hand_landmarks]
    ys = [p.y if hasattr(p, 'y') else p[1] for p in hand_landmarks]

    min_x = min(xs)
    min_y = min(ys)

    features = []
    for p in hand_landmarks:
        px = p.x if hasattr(p, 'x') else p[0]
        py = p.y if hasattr(p, 'y') else p[1]
        features.append(px - min_x)
        features.append(py - min_y)

    return features

def process_dataset():
    # Coloque aqui o fluxo completo de carregamento do MediaPipe Tasks,
    # iteração pelas pastas de frames e geração do dataset/data.pickle
    pass

if __name__ == "__main__":
    process_dataset()