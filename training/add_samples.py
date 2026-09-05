import os
import time
import cv2

DATA_DIR = "dataset/frames/CASA"
os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("\nPosicione as mãos fazendo o sinal de CASA (como no print).")
print("Pressione ESPAÇO na janela de vídeo para gravar 40 fotos.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Sinal: CASA | Pressione ESPACO para gravar", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Captura CASA", frame)
    if cv2.waitKey(1) == 32:
        break

for i in range(40):
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    path = os.path.join(DATA_DIR, f"casa_webcam_{i}_{int(time.time()*1000)}.jpg")
    cv2.imwrite(path, frame)
    cv2.putText(frame, f"Gravando CASA: {i+1}/40", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Captura CASA", frame)
    cv2.waitKey(60)

cap.release()
cv2.destroyAllWindows()
print("Amostras de CASA capturadas com sucesso!")