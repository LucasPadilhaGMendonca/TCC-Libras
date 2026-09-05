import { buildFeatures } from "./lib/features.mjs";
import { SpellingEngine } from "./lib/spelling.mjs";
import { FilesetResolver, HandLandmarker } from "./assets/mediapipe/vision_bundle.mjs";

const LABELS = [
  "A", "B", "C", "D", "E", "F", "G", "I", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"
];

const state = {
  stream: null,
  running: false,
  paused: false,
  landmarker: null,
  session: null,
  busy: false,
  typedText: "",
  lastFrameTime: performance.now(),
  frameCount: 0,
  fps: 0
};

const engine = new SpellingEngine();

// Elementos da interface (compatíveis com o sidepanel.html atualizado)
const video = document.getElementById("webcam");
const btnStart = document.getElementById("btnStart");
const btnStop = document.getElementById("btnStop");
const statusBadge = document.getElementById("statusBadge");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const permissionNotice = document.getElementById("permissionNotice");
const btnOpenPermission = document.getElementById("btnOpenPermission");

const predictedLetter = document.getElementById("predictedLetter");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const fpsCounter = document.getElementById("fpsCounter");

// Elementos opcionais de soletração (caso adicionados ao HTML)
const togglePause = document.getElementById("togglePause");
const sendSpace = document.getElementById("sendSpace");
const sendBackspace = document.getElementById("sendBackspace");
const clearTyped = document.getElementById("clearTyped");
const typedPreview = document.getElementById("typedPreview");
const diagnostics = document.getElementById("diagnostics");

function log(message) {
  if (diagnostics) {
    diagnostics.textContent += message + "\n";
  }
  console.log("[RVL Libras]", message);
}

function updateFPS() {
  const now = performance.now();
  state.frameCount++;

  if (now - state.lastFrameTime >= 1000) {
    state.fps = Math.round((state.frameCount * 1000) / (now - state.lastFrameTime));
    state.frameCount = 0;
    state.lastFrameTime = now;
    if (fpsCounter) {
      fpsCounter.textContent = `${state.fps} FPS`;
    }
  }
}

async function loadModel() {
  if (!window.ort) {
    throw new Error("ONNX Runtime Web não encontrado.");
  }

  window.ort.env.wasm.numThreads = 1;
  window.ort.env.wasm.simd = false;
  window.ort.env.wasm.wasmPaths = chrome.runtime.getURL("assets/onnx/");

  const modelUrl = chrome.runtime.getURL("assets/onnx/model.onnx");

  state.session = await window.ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"]
  });

  log("Modelo ONNX carregado com sucesso.");
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

  log("MediaPipe HandLandmarker carregado.");
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

  const outputs = state.session.outputNames.map(name => ({
    name,
    value: result[name]
  }));

  let labelIndex = null;
  let probabilities = null;

  for (const item of outputs) {
    const val = item.value;

    if (val?.data?.length === 1) {
      const candidate = Number(val.data[0]);
      if (Number.isInteger(candidate)) {
        labelIndex = candidate;
      }
    }

    if (val?.data?.length === LABELS.length) {
      probabilities = Array.from(val.data, Number);
    }
  }

  if (labelIndex === null && probabilities) {
    labelIndex = probabilities.indexOf(Math.max(...probabilities));
  }

  if (labelIndex === null) {
    throw new Error("Não foi possível decodificar saída ONNX.");
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
    return false;
  }
}

function updateTypedPreview() {
  if (typedPreview) {
    typedPreview.textContent = state.typedText.length ? state.typedText : "—";
  }
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
      updateFPS();

      const result = await state.landmarker.detectForVideo(video, performance.now());
      const hand = result?.landmarks?.[0];
      const features = buildFeatures(hand);

      let confirmedLabel = null;

      if (!features) {
        confirmedLabel = engine.processNoHand();
        predictedLetter.textContent = "—";
        confidenceValue.textContent = "0%";
        confidenceBar.style.width = "0%";
      } else {
        const prediction = await predict(features);

        predictedLetter.textContent = prediction.label;

        if (prediction.confidence != null) {
          const pct = Math.min(100, Math.max(0, prediction.confidence * 100));
          confidenceValue.textContent = `${pct.toFixed(0)}%`;
          confidenceBar.style.width = `${pct.toFixed(0)}%`;
        } else {
          confidenceValue.textContent = "—";
          confidenceBar.style.width = "100%";
        }

        confirmedLabel = engine.processDetection(prediction);
      }

      if (confirmedLabel && !state.paused) {
        await typeChar(confirmedLabel);
      }
    } catch (error) {
      log("Erro no frame: " + error.message);
    } finally {
      state.busy = false;
    }
  }

  requestAnimationFrame(processFrame);
}

async function startCamera() {
  if (permissionNotice) {
    permissionNotice.classList.add("hidden");
  }

  statusBadge.textContent = "Iniciando...";
  statusBadge.className = "badge status-standby";

  try {
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

    if (!state.session) {
      statusBadge.textContent = "Carregando IA...";
      await loadModel();
    }

    if (!state.landmarker) {
      statusBadge.textContent = "Carregando Mãos...";
      await loadMediaPipe();
    }

    if (cameraPlaceholder) {
      cameraPlaceholder.classList.add("hidden");
    }

    engine.reset();
    state.typedText = "";
    state.paused = false;
    updateTypedPreview();

    state.running = true;
    btnStart.disabled = true;
    btnStop.disabled = false;

    if (togglePause) togglePause.disabled = false;
    if (sendSpace) sendSpace.disabled = false;
    if (sendBackspace) sendBackspace.disabled = false;
    if (clearTyped) clearTyped.disabled = false;

    statusBadge.textContent = "Ao vivo";
    statusBadge.className = "badge status-active";

    requestAnimationFrame(processFrame);
  } catch (error) {
    log("Erro na câmera: " + error.message);
    statusBadge.textContent = "Inativo";
    statusBadge.className = "badge status-standby";

    // Trata bloqueio de permissão no Sidepanel abrindo uma aba do Chrome para conceder
    if (
      error.name === "NotAllowedError" ||
      error.name === "SecurityError" ||
      error.message.includes("dismissed")
    ) {
      if (permissionNotice) {
        permissionNotice.classList.remove("hidden");
      }
      chrome.tabs.create({ url: chrome.runtime.getURL("sidepanel.html") });
    }
  }
}

function stopCamera() {
  state.running = false;

  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
  }

  video.srcObject = null;
  if (cameraPlaceholder) {
    cameraPlaceholder.classList.remove("hidden");
  }

  btnStart.disabled = false;
  btnStop.disabled = true;

  if (togglePause) togglePause.disabled = true;
  if (sendSpace) sendSpace.disabled = true;
  if (sendBackspace) sendBackspace.disabled = true;
  if (clearTyped) clearTyped.disabled = true;

  predictedLetter.textContent = "—";
  confidenceValue.textContent = "0%";
  confidenceBar.style.width = "0%";
  if (fpsCounter) fpsCounter.textContent = "0 FPS";

  statusBadge.textContent = "Inativo";
  statusBadge.className = "badge status-standby";
}

function onTogglePause() {
  state.paused = !state.paused;
  if (togglePause) {
    togglePause.textContent = state.paused ? "Retomar soletração" : "Pausar soletração";
  }
}

// Event Listeners
btnStart.addEventListener("click", startCamera);
btnStop.addEventListener("click", stopCamera);

if (btnOpenPermission) {
  btnOpenPermission.addEventListener("click", () => {
    chrome.tabs.create({ url: chrome.runtime.getURL("sidepanel.html") });
  });
}

if (togglePause) togglePause.addEventListener("click", onTogglePause);
if (sendSpace) sendSpace.addEventListener("click", () => typeChar(" "));
if (sendBackspace) sendBackspace.addEventListener("click", () => typeBackspace());
if (clearTyped) clearTyped.addEventListener("click", () => clearAllTyped());

log("RVL Libras pronto.");