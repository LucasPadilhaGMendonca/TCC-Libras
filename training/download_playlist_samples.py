import os
import cv2
import yt_dlp

DATA_DIR = "dataset/frames"
TEMP_DIR = "dataset/temp_videos"
os.makedirs(TEMP_DIR, exist_ok=True)

PLAYLIST_MAP = {
    # Aula 01 - Glossário 01: Saudações
    "IoRkuWWliFM": {
        "OI": (1, 3),
        "BOM_DIA": (4, 8),
        "BOA_TARDE": (9, 13),
        "BOA_NOITE": (14, 17),
        "TUDO_BEM": (26, 29),
        "COM_LICENCA": (30, 33),
        "DESCULPA": (34, 38),
        "OBRIGADO": (40, 44),
        "POR_FAVOR": (50, 54),
        "SURDO": (63, 66),
        "OUVINTE": (67, 71),
        "TCHAU": (78, 81),
        "LIBRAS": (88, 92)
    },

    # Aula 01 - Glossário 02: Alfabeto Completo
    "ZeSJQtD3rV0": {
        "A": (1, 4), "B": (5, 8), "C": (9, 12), "D": (17, 19),
        "E": (20, 23), "F": (24, 27), "G": (29, 32), "H": (33, 36),
        "I": (37, 40), "J": (41, 44), "K": (45, 48), "L": (49, 52),
        "M": (53, 56), "N": (57, 60), "O": (63, 66), "P": (68, 71),
        "Q": (72, 75), "R": (77, 80), "S": (82, 85), "T": (86, 89),
        "U": (90, 93), "V": (94, 96), "W": (97, 99), "X": (100, 102),
        "Y": (103, 105), "Z": (106, 109)
    },

    # Aula 01 - Glossário 03: Pronomes (duração total: 112 segundos)
    "vZkXAPsYkA0": {
        "EU": (1, 4),
        "VOCE": (5, 8),
        "ELE_ELA": (9, 12),
        "NOS": (13, 16),
        "VOCES": (17, 20),
        "ELES_ELAS": (21, 24),
        "MEU_MINHA": (74, 78),
        "TEU_TUA": (79, 83),
        "NOSSO": (84, 88),
        "DELE_DELA": (89, 93)
    }
}

def extract_slices(video_path, signs_dict):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    for sign_name, (start_sec, end_sec) in signs_dict.items():
        out_dir = os.path.join(DATA_DIR, sign_name)
        os.makedirs(out_dir, exist_ok=True)

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        total_frames = max(1, end_frame - start_frame)
        step = max(1, total_frames // 30)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        saved = 0
        current = start_frame

        while current <= end_frame and saved < 30:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if (current - start_frame) % step == 0:
                frame_path = os.path.join(out_dir, f"{sign_name}_yt_{saved}.jpg")
                cv2.imwrite(frame_path, frame)
                saved += 1
            current += 1

        print(f"  [+] {sign_name}: {saved} frames gravados em {out_dir}")

    cap.release()

def run():
    # Baixa o fluxo exclusivo de VÍDEO (sem áudio), dispensando totalmente o ffmpeg
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/bestvideo/best',
        'outtmpl': os.path.join(TEMP_DIR, '%(id)s.%(ext)s'),
        'quiet': False,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web']
            }
        }
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for video_id, signs_map in PLAYLIST_MAP.items():
            print(f"\n==========================================")
            print(f"Baixando e processando vídeo: {video_id}...")
            print(f"==========================================")

            url = f"https://www.youtube.com/watch?v={video_id}"
            try:
                info = ydl.extract_info(url, download=True)
                filepath = ydl.prepare_filename(info)

                if not os.path.exists(filepath):
                    base, _ = os.path.splitext(filepath)
                    for f in os.listdir(TEMP_DIR):
                        if f.startswith(os.path.basename(base)):
                            filepath = os.path.join(TEMP_DIR, f)
                            break

                extract_slices(filepath, signs_map)

                if os.path.exists(filepath):
                    os.remove(filepath)

            except Exception as e:
                print(f"Erro ao processar {video_id}: {e}")

    print("\nExtração de todos os glossários finalizada!")

if __name__ == "__main__":
    run()