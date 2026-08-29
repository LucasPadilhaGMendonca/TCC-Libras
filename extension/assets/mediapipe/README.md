Arquivos vendorizados nesta pasta (sem CDN, tudo local):

- `vision_bundle.mjs` — `@mediapipe/tasks-vision@1.0.1` (build ESM), importado
  diretamente por `sidepanel.js` (`type="module"`):
  `import { FilesetResolver, HandLandmarker } from "./assets/mediapipe/vision_bundle.mjs"`.
- `wasm/vision_wasm_internal.js` + `wasm/vision_wasm_internal.wasm` — runtime
  wasm com SIMD (variante padrão do pacote; Chrome e Edge alvo já suportam
  SIMD). Carregado via `FilesetResolver.forVisionTasks(...)`.
- `hand_landmarker.task` — cópia do modelo em `training/hand_landmarker.task`
  (mesmo formato MediaPipe Tasks usado no pipeline Python).

Se quiser atualizar a versão do `@mediapipe/tasks-vision`, baixe os mesmos
arquivos (`vision_bundle.mjs` e a pasta `wasm/`) do pacote npm na versão
desejada.
