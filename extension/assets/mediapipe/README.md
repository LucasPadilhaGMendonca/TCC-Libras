Coloque aqui os arquivos locais do MediaPipe Tasks Vision.

A extensão espera um `vision_bundle.js` que exponha:

```javascript
window.SignForestVision.createHandLandmarker()
```

A função deve retornar um Hand Landmarker configurado para detectar uma única mão.

No build final do TCC, mantenha os arquivos localmente dentro da extensão,
sem depender de CDN/código remoto.
