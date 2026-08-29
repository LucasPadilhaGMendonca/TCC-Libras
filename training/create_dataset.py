import pickle
from pathlib import Path
import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[1]
FRAME_DIR = ROOT / "dataset" / "frames"
OUT_FILE = ROOT / "dataset" / "data.pickle"

mp_hands = mp.solutions.hands

def build_features(hand_landmarks):
    """
    Normaliza 21 landmarks (x, y) subtraindo os menores valores,
    resultando em 42 features.
    """
    data_aux = []
    x_ = [landmark.x for landmark in hand_landmarks.landmark]
    y_ = [landmark.y for landmark in hand_landmarks.landmark]

    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))

    return data_aux if len(data_aux) == 42 else None

def main():
    if not FRAME_DIR.exists():
        raise SystemExit(
            "dataset/frames não encontrado. Execute primeiro: python training/extract_frames.py"
        )

    data = []
    labels = []
    total_frames = 0
    valid_frames = 0
    no_hand = 0
    multiple_hands = 0

    class_dirs = [p for p in sorted(FRAME_DIR.iterdir()) if p.is_dir()]

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    ) as hands:
        for class_dir in class_dirs:
            class_label = class_dir.name
            print(f"\n=== Processando classe: {class_label} ===")

            for image_path in sorted(class_dir.glob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue

                total_frames += 1
                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if not results.multi_hand_landmarks:
                    no_hand += 1
                    continue

                if len(results.multi_hand_landmarks) != 1:
                    multiple_hands += 1
                    continue

                features = build_features(results.multi_hand_landmarks[0])
                if features is not None:
                    data.append(features)
                    labels.append(class_label)
                    valid_frames += 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)

    print("\n==============================")
    print(f"Frames analisados:    {total_frames}")
    print(f"Frames válidos:       {valid_frames}")
    print(f"Sem mão detectada:    {no_hand}")
    print(f"Mais de uma mão:      {multiple_hands}")
    print(f"Arquivo gerado:       {OUT_FILE}")
    print("==============================")

if __name__ == "__main__":
    main()