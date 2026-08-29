Arquivos vendorizados nesta pasta (sem CDN, tudo local):

- `ort.min.js` + `ort-wasm.wasm` — ONNX Runtime Web `1.14.0`, build wasm sem
  SIMD/threads (`ort-wasm.wasm`), suficiente para um modelo RandomForest
  pequeno como este. Baixados de `onnxruntime-web@1.14.0` no npm.
- `labels.json` — as 20 classes do modelo, na mesma ordem usada pelo índice
  de saída do RandomForest exportado.
- `model.onnx` — **não versionado** (ver `.gitignore`). Gere localmente com:

```powershell
python training/create_dataset.py
python training/train_classifier.py
python training/export_model.py
```

A ordem de features esperada pelo modelo é exatamente:
[x0-minX, y0-minY, x1-minX, y1-minY, ..., x20-minX, y20-minY], total 42 valores.

Se quiser atualizar a versão do ONNX Runtime Web vendorizada, baixe os
mesmos dois arquivos (`dist/ort.min.js` e `dist/ort-wasm.wasm`) do pacote
npm `onnxruntime-web` na versão desejada e ajuste a referência de versão
aqui.
