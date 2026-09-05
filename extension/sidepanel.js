import { buildFeatures } from "./lib/features.mjs";
import { SpellingEngine } from "./lib/spelling.mjs";
import { FilesetResolver, HandLandmarker } from "./assets/mediapipe/vision_bundle.mjs";

let LABELS = [];

const state = {
  stream: null,
  running: false,
  landmarker: null,
  session: null,
  busy: false,
  typedText: "",
  lastFrameTime: performance.now(),
  frameCount: 0,
  fps: 0
};

const engine = new SpellingEngine({
  stableFramesRequired: 6,
  confidenceThreshold: 0.40,
  releaseFramesRequired: 3
});

// Elementos DOM
const video = document.getElementById("webcam");
const btnStart = document.getElementById("btnStart");
const btnStop = document.getElementById("btnStop");
const statusBadge = document.getElementById("statusBadge");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const permissionNotice = document.getElementById("permissionNotice");
const btnOpenPermission = document.getElementById("btnOpenPermission");

const predictedLetter = document.getElementById("predictedLetter");
const confidenceValue = document.getElementById("confidenceValue");
const stabilityFill = document.getElementById("stabilityFill");
const fpsCounter = document.getElementById("fpsCounter");

const typedPreview = document.getElementById("typedPreview");
const btnSpace = document.getElementById("btnSpace");
const btnBackspace = document.getElementById("btnBackspace");
const btnClear = document.getElementById("btnClear");
const btnSearch = document.getElementById("btnSearch");

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

async function loadLabels() {
  try {
    const res = await fetch(chrome.runtime.getURL("assets/onnx/labels.json"));
    if (res.ok) {
      LABELS = await res.json();
      console.log("[RVL Libras] Labels carregadas:", LABELS);
      return;
    }
  } catch (_) {}

  LABELS = [
    "A", "AMOR", "B", "C", "CASA", "D", "E", "F", "G", "I",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"
  ];
}

async function loadModel() {
  if (!window.ort) {
    throw new Error("ort.min.js não carregado.");
  }

  window.ort.env.wasm.numThreads = 1;
  window.ort.env.wasm.simd = false;
  window.ort.env.wasm.proxy = false;
  window.ort.env.wasm.wasmPaths = chrome.runtime.getURL("assets/onnx/");

  const modelUrl = chrome.runtime.getURL("assets/onnx/model.onnx");
  state.session = await window.ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "disabled"
  });
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
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
}

async function predict(features) {
  if (!state.session || !features || features.length !== 84) return null;

  const inputName = state.session.inputNames[0];
  const tensor = new window.ort.Tensor("float32", Float32Array.from(features), [1, 84]);
  const result = await state.session.run({ [inputName]: tensor });

  let probabilities = null;

  for (const name of state.session.outputNames) {
    const val = result[name];
    if (val?.data?.length === LABELS.length) {
      probabilities = Array.from(val.data, Number);
      break;
    }
  }

  if (probabilities) {
    const maxProb = Math.max(...probabilities);
    const labelIndex = probabilities.indexOf(maxProb);

    // Limiar balanceado para 51 classes: 40%
    if (labelIndex === -1 || maxProb < 0.40) {
      return null;
    }

    return {
      label: LABELS[labelIndex] ?? String(labelIndex),
      confidence: maxProb
    };
  }

  return null;
}

async function sendToContentScript(action, value = "") {
  try {
    const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (activeTab?.id) {
      await chrome.tabs.sendMessage(activeTab.id, {
        type: "SIGNFOREST_KEY",
        action,
        value
      });
    }
  } catch (_) {}
}

function updateTypedDisplay() {
  typedPreview.textContent = state.typedText.length > 0 ? state.typedText : "—";
}

function appendText(text) {
  const toAdd = text.length > 1 ? `${text} ` : text;
  state.typedText += toAdd;
  updateTypedDisplay();
  sendToContentScript("char", toAdd);
}

function handleBackspace() {
  if (state.typedText.length > 0) {
    state.typedText = state.typedText.trimEnd().slice(0, -1);
    updateTypedDisplay();
    sendToContentScript("backspace");
  }
}

function handleClear() {
  state.typedText = "";
  updateTypedDisplay();
}

function handleSearch() {
  const query = state.typedText.trim();
  if (query.length > 0) {
    chrome.tabs.create({ url: `https://www.google.com/search?q=${encodeURIComponent(query)}` });
  }
}

async function processFrame() {
  if (!state.running) return;

  updateFPS();

  if (!state.busy && state.landmarker && state.session) {
    state.busy = true;

    try {
      const result = await state.landmarker.detectForVideo(video, performance.now());
      const hands = result?.landmarks ?? [];

      if (hands.length === 0) {
        engine.processNoHand();
        predictedLetter.textContent = "—";
        stabilityFill.style.width = "0%";
        confidenceValue.textContent = "0%";
      } else {
        const features = buildFeatures(hands);
        const prediction = features ? await predict(features) : null;

        if (prediction) {
          predictedLetter.textContent = prediction.label;
          const confirmedWord = engine.processDetection(prediction);

          const progressPercent = Math.min(100, Math.round(engine.progress * 100));
          stabilityFill.style.width = `${progressPercent}%`;
          confidenceValue.textContent = `${Math.round(prediction.confidence * 100)}%`;

          if (confirmedWord) {
            appendText(confirmedWord);
          }
        } else {
          engine.processNoHand();
          predictedLetter.textContent = "—";
          stabilityFill.style.width = "0%";
          confidenceValue.textContent = "0%";
        }
      }
    } catch (err) {
      console.warn("Erro no processamento do frame:", err);
    } finally {
      state.busy = false;
    }
  }

  requestAnimationFrame(processFrame);
}

async function startCamera() {
  if (permissionNotice) permissionNotice.classList.add("hidden");

  statusBadge.textContent = "Iniciando...";
  statusBadge.className = "badge status-standby";

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false
    });

    video.srcObject = state.stream;
    await video.play();

    if (cameraPlaceholder) cameraPlaceholder.classList.add("hidden");
    btnStart.disabled = true;
    btnStop.disabled = false;

    statusBadge.textContent = "Carregando IA...";
    await loadLabels();
    if (!state.session) await loadModel();
    if (!state.landmarker) await loadMediaPipe();

    engine.reset();
    state.running = true;
    state.lastFrameTime = performance.now();
    state.frameCount = 0;
    requestAnimationFrame(processFrame);

    statusBadge.textContent = "Ao vivo";
    statusBadge.className = "badge status-active";
  } catch (error) {
    statusBadge.textContent = "Inativo";
    statusBadge.className = "badge status-standby";
    btnStart.disabled = false;
    btnStop.disabled = true;
    state.running = false;

    if (state.stream) {
      state.stream.getTracks().forEach(t => t.stop());
      state.stream = null;
    }

    if (
      error.name === "NotAllowedError" ||
      error.name === "SecurityError" ||
      error.message?.includes("dismissed")
    ) {
      if (permissionNotice) permissionNotice.classList.remove("hidden");
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
  if (cameraPlaceholder) cameraPlaceholder.classList.remove("hidden");

  btnStart.disabled = false;
  btnStop.disabled = true;

  predictedLetter.textContent = "—";
  stabilityFill.style.width = "0%";
  confidenceValue.textContent = "0%";
  if (fpsCounter) fpsCounter.textContent = "0 FPS";

  statusBadge.textContent = "Inativo";
  statusBadge.className = "badge status-standby";
}

btnStart.addEventListener("click", startCamera);
btnStop.addEventListener("click", stopCamera);

btnSpace.addEventListener("click", () => appendText(" "));
btnBackspace.addEventListener("click", handleBackspace);
btnClear.addEventListener("click", handleClear);
btnSearch.addEventListener("click", handleSearch);

if (btnOpenPermission) {
  btnOpenPermission.addEventListener("click", () => {
    chrome.tabs.create({ url: chrome.runtime.getURL("sidepanel.html") });
  });
}