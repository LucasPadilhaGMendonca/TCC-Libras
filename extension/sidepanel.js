import { buildFeatures } from "./lib/features.mjs";
import { SpellingEngine } from "./lib/spelling.mjs";
import { FilesetResolver, HandLandmarker } from "./assets/mediapipe/vision_bundle.mjs";

const LABELS = [
  "A","B","C","D","E","F","G","I","L","M",
  "N","O","P","Q","R","S","T","U","V","W"
];

const state = {
  stream: null,
  running: false,
  paused: false,
  landmarker: null,
  session: null,
  busy: false,
  typedText: ""
};

const engine = new SpellingEngine();

const video = document.getElementById("video");
const start = document.getElementById("startCamera");
const stop = document.getElementById("stopCamera");
const togglePause = document.getElementById("togglePause");
const sendSpace = document.getElementById("sendSpace");
const sendBackspace = document.getElementById("sendBackspace");
const clearTyped = document.getElementById("clearTyped");
const status = document.getElementById("status");
const text = document.getElementById("recognizedText");
const confidence = document.getElementById("confidence");
const diagnostics = document.getElementById("diagnostics");
const stabilityFill = document.getElementById("stabilityFill");
const typedPreview = document.getElementById("typedPreview");

function log(message) {
  diagnostics.textContent += message + "\n";
}

async function loadModel() {
  if (!window.ort) {
    throw new Error("ONNX Runtime Web não encontrado.");
  }

  // Usamos apenas o binário wasm sem SIMD/threads (assets/onnx/ort-wasm.wasm)
  // para manter um único arquivo vendorizado; suficiente para um modelo
  // RandomForest pequeno como este.
  window.ort.env.wasm.numThreads = 1;
  window.ort.env.wasm.simd = false;
  window.ort.env.wasm.wasmPaths = chrome.runtime.getURL("assets/onnx/");

  const modelUrl = chrome.runtime.getURL("assets/onnx/model.onnx");

  state.session = await window.ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"]
  });

  log("Modelo ONNX carregado.");
  log("Entrada: " + state.session.inputNames.join(", "));
  log("Saída: " + state.session.outputNames.join(", "));
}

async function loadMediaPipe() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    chrome.runtime.getURL("assets/mediapipe/wasm")
  );

  state.landmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: chrome.runtime.getURL("assets/mediapipe/hand_landmarker.task")
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.3
  });

  log("MediaPipe carregado.");
}

async function predict(features) {
  const inputName = state.session.inputNames[0];

  const tensor = new window.ort.Tensor(
    "float32",
    Float32Array.from(features),
    [1, 42]
  );

  const result = await state.session.run({
    [inputName]: tensor
  });

  // sklearn/RandomForest exportado pelo skl2onnx normalmente fornece:
  // - output de label
  // - output de probabilidades quando zipmap=False.
  const outputs = state.session.outputNames.map(name => ({
    name,
    value: result[name]
  }));

  let labelIndex = null;
  let probabilities = null;

  for (const item of outputs) {
    const value = item.value;

    if (value?.data?.length === 1) {
      const candidate = Number(value.data[0]);
      if (Number.isInteger(candidate)) {
        labelIndex = candidate;
      }
    }

    if (value?.data?.length === LABELS.length) {
      probabilities = Array.from(value.data, Number);
    }
  }

  if (labelIndex === null && probabilities) {
    labelIndex = probabilities.indexOf(Math.max(...probabilities));
  }

  if (labelIndex === null) {
    throw new Error("Não foi possível interpretar a saída do ONNX.");
  }

  const score = probabilities ? probabilities[labelIndex] : null;

  return {
    label: LABELS[labelIndex] ?? String(labelIndex),
    confidence: score
  };
}

async function sendKey(action, value) {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) return false;

    const response = await chrome.tabs.sendMessage(tab.id, {
      type: "SIGNFOREST_KEY",
      action,
      value
    });

    return Boolean(response?.ok);
  } catch (error) {
    log("Content Script: " + error.message);
    return false;
  }
}

function updateTypedPreview() {
  typedPreview.textContent = state.typedText.length ? state.typedText : "—";
}

async function typeChar(char) {
  const ok = await sendKey("char", char);
  if (ok) {
    state.typedText += char;
    updateTypedPreview();
  }
}

async function typeBackspace() {
  const ok = await sendKey("backspace");
  if (ok && state.typedText.length) {
    state.typedText = state.typedText.slice(0, -1);
    updateTypedPreview();
  }
}

async function clearAllTyped() {
  const count = state.typedText.length;
  for (let i = 0; i < count; i++) {
    await sendKey("backspace");
  }
  state.typedText = "";
  updateTypedPreview();
}

async function processFrame() {
  if (!state.running) return;

  if (!state.busy) {
    state.busy = true;

    try {
      const result = await state.landmarker.detectForVideo(video, performance.now());
      const hand = result?.landmarks?.[0];
      const features = buildFeatures(hand);

      let confirmedLabel = null;

      if (!features) {
        confirmedLabel = engine.processNoHand();
        text.textContent = "—";
        confidence.textContent = "Confiança: —";
      } else {
        const prediction = await predict(features);

        text.textContent = prediction.label;
        confidence.textContent =
          prediction.confidence == null
            ? "Confiança: indisponível"
            : `Confiança: ${(prediction.confidence * 100).toFixed(1)}%`;

        confirmedLabel = engine.processDetection(prediction);
      }

      stabilityFill.style.width = `${(engine.progress * 100).toFixed(0)}%`;

      if (confirmedLabel && !state.paused) {
        await typeChar(confirmedLabel);
      }
    } catch (error) {
      log("Frame: " + error.message);
    } finally {
      state.busy = false;
    }
  }

  requestAnimationFrame(processFrame);
}

async function startCamera() {
  try {
    status.textContent = "Inicializando...";

    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 640,
        height: 480,
        facingMode: "user"
      },
      audio: false
    });

    video.srcObject = state.stream;
    await video.play();

    await loadModel();
    await loadMediaPipe();

    engine.reset();
    state.typedText = "";
    state.paused = false;
    updateTypedPreview();
    togglePause.textContent = "Pausar soletração";

    state.running = true;
    start.disabled = true;
    stop.disabled = false;
    togglePause.disabled = false;
    sendSpace.disabled = false;
    sendBackspace.disabled = false;
    clearTyped.disabled = false;
    status.textContent = "Soletração automática ativa";

    requestAnimationFrame(processFrame);
  } catch (error) {
    status.textContent = "Erro: " + error.message;
    log(error.stack || error.message);
  }
}

function stopCamera() {
  state.running = false;

  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
  }

  video.srcObject = null;
  start.disabled = false;
  stop.disabled = true;
  togglePause.disabled = true;
  sendSpace.disabled = true;
  sendBackspace.disabled = true;
  clearTyped.disabled = true;
  stabilityFill.style.width = "0%";
  status.textContent = "Câmera parada";
}

function onTogglePause() {
  state.paused = !state.paused;
  togglePause.textContent = state.paused ? "Retomar soletração" : "Pausar soletração";
  status.textContent = state.paused ? "Soletração pausada" : "Soletração automática ativa";
}

start.addEventListener("click", startCamera);
stop.addEventListener("click", stopCamera);
togglePause.addEventListener("click", onTogglePause);
sendSpace.addEventListener("click", () => typeChar(" "));
sendBackspace.addEventListener("click", () => typeBackspace());
clearTyped.addEventListener("click", () => clearAllTyped());

log("SignForest carregado.");
