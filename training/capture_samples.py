import os
import time
import cv2

DATA_DIR = "dataset/frames"
CLASSES = ["AMOR", "CASA", "BRASIL"]
SAMPLES_PER_CLASS = 30  # Quantidade de fotos por sinal

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

for class_name in CLASSES:
    target_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n==========================================")
    print(f"PREPARE-SE PARA O SINAL: [{class_name}]")
    print(f"Posicione as duas mãos em frente à câmera.")
    print(f"Aperte 'ESPAÇO' na janela de vídeo para iniciar.")
    print(f"==========================================")

    # Espera o usuário se posicionar e apertar ESPAÇO
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame, f"Sinal: {class_name} | Aperte ESPACO para gravar",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.imshow("Captura RVL Libras", frame)

        key = cv2.waitKey(1)
        if key == 32:  # Barra de espaço
            break
        elif key == 27:  # ESC cancela
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Contagem regressiva antes da rajada de capturas
    for countdown in [3, 2, 1]:
        start_t = time.time()
        while time.time() - start_t < 1.0:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame, f"Gravando em: {countdown}...",
                (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3
            )
            cv2.imshow("Captura RVL Libras", frame)
            cv2.waitKey(1)

    # Captura as fotos consecutivas variando levemente a posição
    print(f"Capturando {SAMPLES_PER_CLASS} amostras de {class_name}...")
    captured = 0
    while captured < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        img_path = os.path.join(target_dir, f"{class_name}_{captured}_{int(time.time()*1000)}.jpg")
        cv2.imwrite(img_path, frame)

        cv2.putText(
            frame, f"Gravando [{class_name}]: {captured + 1}/{SAMPLES_PER_CLASS}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
        cv2.imshow("Captura RVL Libras", frame)
        cv2.waitKey(80)  # Intervalo de 80ms entre capturas
        captured += 1

print("\nTodas as amostras foram gravadas com sucesso!")
cap.release()
cv2.destroyAllWindows()