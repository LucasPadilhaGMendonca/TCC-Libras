const LABELS = [
  "A","B","C","D","E","F","G","I","L","M",
  "N","O","P","Q","R","S","T","U","V","W"
];

const state = {
  stream: null,
  running: false,
  landmarker: null,
  session: null,
  prediction: null,
  busy: false
};

const video = document.getElementById("video");
const start = document.getElementById("startCamera");
const stop = document.getElementById("stopCamera");
const send = document.getElementById("sendToPage");
const status = document.getElementById("status");
const text = document.getElementById("recognizedText");
const confidence = document.getElementById("confidence");
const diagnostics = document.getElementById("diagnostics");

function log(message) {
  diagnostics.textContent += message + "\n";
}

function buildFeatures(hand) {
  // EXATAMENTE a lógica do projeto Python anterior:
  // x_i - min(x), y_i - min(y), total = 42 features.
  if (!hand || hand.length !== 21) return null;

  const xs = hand.map(p => p.x);
  const ys = hand.map(p => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);

  const features = [];

  for (const p of hand) {
    features.push(p.x - minX);
    features.push(p.y - minY);
  }

  return features.length === 42 ? features : null;
}

async function loadModel() {
  if (!window.ort) {
    throw new Error("ONNX Runtime Web não encontrado.");
  }

  const modelUrl = chrome.runtime.getURL("assets/onnx/model.onnx");

  state.session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"]
  });

  log("Modelo ONNX carregado.");
  log("Entrada: " + state.session.inputNames.join(", "));
  log("Saída: " + state.session.outputNames.join(", "));
}

async function loadMediaPipe() {
  if (!window.SignForestVision) {
    throw new Error(
      "MediaPipe local não configurado. Veja assets/mediapipe/README.md."
    );
  }

  state.landmarker =
    await window.SignForestVision.createHandLandmarker();

  log("MediaPipe carregado.");
}

async function predict(features) {
  const inputName = state.session.inputNames[0];

  const tensor = new ort.Tensor(
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

  const score = probabilities
    ? probabilities[labelIndex]
    : null;

  return {
    label: LABELS[labelIndex] ?? String(labelIndex),
    confidence: score
  };
}

async function processFrame() {
  if (!state.running) return;

  // Evita acumular inferências assíncronas.
  if (!state.busy) {
    state.busy = true;

    try {
      const result = await state.landmarker.detectForVideo(
        video,
        performance.now()
      );

      const hand = result?.landmarks?.[0];
      const features = buildFeatures(hand);

      if (features) {
        const prediction = await predict(features);

        state.prediction = prediction;
        text.textContent = prediction.label;

        confidence.textContent =
          prediction.confidence == null
            ? "Confiança: disponível somente após ajuste da saída"
            : `Confiança: ${(prediction.confidence * 100).toFixed(1)}%`;

        send.disabled = false;
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

    state.running = true;
    start.disabled = true;
    stop.disabled = false;
    status.textContent = "Reconhecimento ativo";

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
  status.textContent = "Câmera parada";
}

async function sendToPage() {
  if (!state.prediction) return;

  const [tab] = await chrome.tabs.query({
    active: true,
    currentWindow: true
  });

  if (!tab?.id) return;

  chrome.tabs.sendMessage(tab.id, {
    type: "SIGNFOREST_INSERT_TEXT",
    text: state.prediction.label
  }).catch(error => log("Content Script: " + error.message));
}

start.addEventListener("click", startCamera);
stop.addEventListener("click", stopCamera);
send.addEventListener("click", sendToPage);

log("SignForest carregado.");
