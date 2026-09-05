import os
import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "frames")
OUTPUT_PICKLE = os.path.join(os.path.dirname(__file__), "..", "dataset", "data.pickle")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "extension", "assets", "mediapipe", "hand_landmarker.task")


def build_features(hand_landmarks):
    """
    Normaliza os marcos de uma mão em relação ao bounding box mínimo (minX, minY).
    Retorna 42 floats para 21 pontos ou None caso não contenha exatamente 21 marcos.
    """
    if not hand_landmarks or len(hand_landmarks) != 21:
        return None

    xs = [p.x if hasattr(p, "x") else p[0] for p in hand_landmarks]
    ys = [p.y if hasattr(p, "y") else p[1] for p in hand_landmarks]

    min_x = min(xs)
    min_y = min(ys)

    features = []
    for p in hand_landmarks:
        px = p.x if hasattr(p, "x") else p[0]
        py = p.y if hasattr(p, "y") else p[1]
        features.append(px - min_x)
        features.append(py - min_y)

    return features


def process_dataset():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    data = []
    labels = []

    total_frames = 0
    valid_frames = 0
    no_hand_frames = 0

    print("Iniciando extração bimanual (84 features com Tasks API)...")

    with vision.HandLandmarker.create_from_options(options) as detector:
        if not os.path.exists(DATA_DIR):
            print(f"Diretório {DATA_DIR} não encontrado.")
            return

        classes = sorted(os.listdir(DATA_DIR))

        for dir_ in classes:
            class_path = os.path.join(DATA_DIR, dir_)
            if not os.path.isdir(class_path):
                continue

            print(f"=== Processando classe: {dir_} ===")

            for img_path in sorted(os.listdir(class_path)):
                if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                total_frames += 1
                full_img_path = os.path.join(class_path, img_path)

                cv_img = cv2.imread(full_img_path)
                if cv_img is None:
                    continue

                rgb_frame = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                detection_result = detector.detect(mp_image)

                if detection_result.hand_landmarks:
                    hand1 = detection_result.hand_landmarks[0]
                    hand2 = detection_result.hand_landmarks[1] if len(detection_result.hand_landmarks) > 1 else None

                    f1 = build_features(hand1)
                    if f1 is None:
                        continue

                    f2 = build_features(hand2) if hand2 is not None else [0.0] * 42

                    combined = f1 + f2
                    if len(combined) == 84:
                        data.append(combined)
                        labels.append(dir_)
                        valid_frames += 1
                else:
                    no_hand_frames += 1

    os.makedirs(os.path.dirname(OUTPUT_PICKLE), exist_ok=True)
    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)

    print("\n==============================")
    print(f"Frames analisados:    {total_frames}")
    print(f"Frames válidos:       {valid_frames}")
    print(f"Sem mão detectada:    {no_hand_frames}")
    print(f"Features por amostra: 84")
    print(f"Arquivo gerado:       {OUTPUT_PICKLE}")
    print("==============================")


if __name__ == "__main__":
    process_dataset()