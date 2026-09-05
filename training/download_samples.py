import os
import cv2
import yt_dlp

DATA_DIR = "dataset/frames"

# O yt-dlp busca automaticamente o melhor vídeo no YouTube para o termo
SEARCH_QUERIES = {
    "AMOR": "ytsearch1:sinal libras amor",
    "CASA": "ytsearch1:sinal libras casa",
    "BRASIL": "ytsearch1:sinal libras brasil"
}

TEMP_VIDEO_DIR = "dataset/temp_videos"
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': os.path.join(TEMP_VIDEO_DIR, '%(id)s.%(ext)s'),
    'quiet': False,
    'no_warnings': True,
    'ignoreerrors': True
}

def extract_frames(video_path, output_dir, label, max_frames=35):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return

    step = max(1, total // (max_frames + 4))
    count = 0
    saved = 0
    
    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % step == 0 and count > step:
            frame_path = os.path.join(output_dir, f"{label}_yt_{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
            
        count += 1
        
    cap.release()
    print(f"[{label}] -> {saved} frames extraídos e salvos em {output_dir}")

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for label, query in SEARCH_QUERIES.items():
        print(f"\n==========================================")
        print(f"Buscando e baixando vídeo no YouTube para: {label}...")
        print(f"==========================================")
        try:
            info = ydl.extract_info(query, download=True)
            if not info or 'entries' not in info or not info['entries']:
                print(f"Não foi possível encontrar vídeo para {label}.")
                continue
                
            entry = info['entries'][0]
            video_file = ydl.prepare_filename(entry)
            
            # Se o formato final mudou de extensão (ex: mkv)
            if not os.path.exists(video_file):
                base, _ = os.path.splitext(video_file)
                for f in os.listdir(TEMP_VIDEO_DIR):
                    if f.startswith(os.path.basename(base)):
                        video_file = os.path.join(TEMP_VIDEO_DIR, f)
                        break

            target_dir = os.path.join(DATA_DIR, label)
            extract_frames(video_file, target_dir, label)
            
            if os.path.exists(video_file):
                os.remove(video_file)
        except Exception as e:
            print(f"Erro ao processar {label}: {e}")

print("\nProcesso finalizado!")