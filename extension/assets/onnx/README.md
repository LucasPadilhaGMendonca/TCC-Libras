Após executar:

```powershell
python training/create_dataset.py
python training/train_classifier.py
python training/export_model.py
```

o arquivo `model.onnx` aparecerá nesta pasta.

Também coloque aqui:
- `ort.min.js`
- os arquivos WASM do ONNX Runtime Web

A ordem de features esperada pelo modelo é exatamente:
[x0-minX, y0-minY, x1-minX, y1-minY, ..., x20-minX, y20-minY]
total: 42 valores.
