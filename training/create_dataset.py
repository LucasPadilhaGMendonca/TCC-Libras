import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIR = "dataset/frames"
OUTPUT_FILE = "dataset/data.pickle"
MODEL_PATH = "training/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    # Fallback para o caminho da extensão se não estiver em training
    MODEL_PATH = "extension/assets/mediapipe/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Arquivo de modelo não encontrado: {MODEL_PATH}")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3
)

landmarker = vision.HandLandmarker.create_from_options(options)

data = []
labels = []

total_frames = 0
valid_frames = 0
no_hand_frames = 0

def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y])
    
    # Normalização em relação ao ponto do pulso (landmark 0)
    base_x = coords[0]
    base_y = coords[1]
    norm_coords = []
    for i in range(0, len(coords), 2):
        norm_coords.append(coords[i] - base_x)
        norm_coords.append(coords[i+1] - base_y)
    return norm_coords

print("Iniciando extração bimanual (84 features com Tasks API)...")

for dir_ in sorted(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f"=== Processando classe: {dir_} ===")

    for img_path in os.listdir(dir_path):
        if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        total_frames += 1
        full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(full_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        results = landmarker.detect(mp_image)

        if not results.hand_landmarks:
            no_hand_frames += 1
            continue

        hand_list = results.hand_landmarks
        combined_features = []

        # Mão 1 (42 features)
        combined_features.extend(extract_landmarks(hand_list[0]))

        # Mão 2 (42 features) - completa com zeros se houver só 1 mão
        if len(hand_list) > 1:
            combined_features.extend(extract_landmarks(hand_list[1]))
        else:
            combined_features.extend([0.0] * 42)

        if len(combined_features) == 84:
            data.append(combined_features)
            labels.append(dir_)
            valid_frames += 1

print("\n" + "=" * 30)
print(f"Frames analisados:    {total_frames}")
print(f"Frames válidos:       {valid_frames}")
print(f"Sem mão detectada:    {no_hand_frames}")
print(f"Features por amostra: 84")
print(f"Arquivo gerado:       {os.path.abspath(OUTPUT_FILE)}")
print("=" * 30)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

landmarker.close()