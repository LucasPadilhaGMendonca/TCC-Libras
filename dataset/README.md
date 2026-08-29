Estrutura real usada pelo pipeline (ver README raiz):

```
dataset/videos/<CLASSE>/*.mp4   → vídeos brutos, um ou mais por classe
dataset/frames/<CLASSE>/*.jpg   → gerado por training/extract_frames.py
dataset/data.pickle             → gerado por training/create_dataset.py
```

Classes cobertas hoje (alfabeto datilológico, apenas sinais **estáticos**):
A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W

As letras H, J, K, X, Y, Z ficam de fora porque seus sinais em LIBRAS
envolvem movimento, e o pipeline atual (MediaPipe Hands sobre frames
isolados) só reconhece poses estáticas de mão — não sequências.

Fluxo para adicionar uma classe/vídeo novo:

1. Coloque o(s) vídeo(s) em `dataset/videos/<CLASSE>/`.
2. `python training/extract_frames.py`
3. `python training/create_dataset.py`
4. `python training/train_classifier.py`
5. `python training/export_model.py`

Para que a acurácia reportada por `train_classifier.py` seja confiável,
use mais de um vídeo/intérprete por classe — com um único vídeo por
classe, o script cai automaticamente para um split aleatório por frame,
que tende a superestimar a acurácia (frames do mesmo vídeo são muito
correlacionados entre si).
