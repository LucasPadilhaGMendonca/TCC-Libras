import os
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "dataset" / "videos"
FRAME_DIR = ROOT / "dataset" / "frames"

# Quantidade de frames extraídos por segundo do vídeo
FRAME_SAMPLE_FPS = 8
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

def extract_video(video_path: Path, output_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print(f"[ERRO] FPS inválido: {video_path}")
        cap.release()
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames else 0
    step = max(1, int(round(fps / FRAME_SAMPLE_FPS)))

    saved = 0
    frame_index = 0
    video_name = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step == 0:
            filename = f"{video_name}_frame_{frame_index:07d}.jpg"
            output_path = output_dir / filename
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1

        frame_index += 1

    cap.release()
    print(f"[OK] {video_path.name} | {duration:.1f}s | {saved} frames")
    return saved

def main():
    if not VIDEO_DIR.exists():
        raise SystemExit(f"Pasta não encontrada: {VIDEO_DIR}")

    total_videos = 0
    total_frames = 0

    for class_dir in sorted(VIDEO_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        output_dir = FRAME_DIR / class_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        videos = [
            p for p in sorted(class_dir.iterdir())
            if p.suffix.lower() in VIDEO_EXTENSIONS
        ]

        if not videos:
            continue

        print(f"\n=== Classe: {class_dir.name} ===")
        for video_path in videos:
            total_videos += 1
            total_frames += extract_video(video_path, output_dir)

    print("\n==============================")
    print(f"Vídeos processados: {total_videos}")
    print(f"Frames gerados:     {total_frames}")
    print(f"Saída:              {FRAME_DIR}")
    print("==============================")

if __name__ == "__main__":
    main()