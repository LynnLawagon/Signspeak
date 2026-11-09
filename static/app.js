/*
 * SignSpeak — in-browser sign recognition.
 *
 * Everything runs on the visitor's device:
 *   1. MediaPipe Hands (same hand model cvzone uses) finds the hand and gives a bounding box.
 *   2. The hand is cropped and prepared EXACTLY like datacollection.py / test.py did when the
 *      Teachable Machine models were trained (mirrored frame, skeleton + pink box drawn on the
 *      frame, 20px offset, hand placed on a 300x300 white square).
 *   3. TensorFlow.js runs the converted Teachable Machine model (static/models/<mode>/).
 *
 * The server only serves files, so a small Render instance is enough.
 */
"use strict";

const video = document.getElementById("camera");
const frameCanvas = document.getElementById("capture");          // mirrored camera frame (hidden)
const frameCtx = frameCanvas.getContext("2d", { willReadFrequently: false });
const overlay = document.getElementById("handOverlay");
const overlayCtx = overlay.getContext("2d");
const startButton = document.getElementById("startCamera");
const stopButton = document.getElementById("stopCamera");
const cameraStatus = document.getElementById("cameraStatus");
const placeholder = document.getElementById("cameraPlaceholder");
const cameraHint = document.querySelector(".camera-hint");
const resultSymbol = document.getElementById("resultSymbol");
const resultLabel = document.getElementById("resultLabel");
const resultDescription = document.getElementById("resultDescription");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const modeButtons = document.querySelectorAll(".mode-tab");
const liveDot = document.querySelector(".live-indicator i");

// ---- Settings (kept identical to datacollection.py / test.py) -------------------------------
const OFFSET = 20;            // padding around the hand, in pixels of a 640px-wide frame
const IMG_SIZE = 300;         // hand is placed on a 300x300 white square
const MODEL_SIZE = 224;       // Teachable Machine input size
const FRAME_WIDTH = 640;      // camera frames are processed at (at most) this width
const CLASSIFY_EVERY_MS = 120;
const SMOOTHING_FRAMES = 5;   // average the last N predictions so the label does not flicker
const MIN_CONFIDENCE = 0.40;  // below this the result panel says "not sure yet"
const OFFENSIVE_LABELS = new Set(["f*** you"]);

const BASE = new URL("static/", document.baseURI).href;
const MODEL_URLS = {
  words: BASE + "models/words/model.json",
  letters: BASE + "models/letters/model.json",
  numbers: BASE + "models/numbers/model.json"
};

// Same connections as MediaPipe's HAND_CONNECTIONS (used to draw the training-style skeleton).
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

const descriptions = {
  "drink": "To take in liquid refreshment.",
  "f*** you": "An offensive or rude gesture.",
  "food": "Something people eat for nourishment.",
  "hello": "A greeting used when meeting someone.",
  "i love you": "Words expressing affection or love.",
  "no": "Used to refuse or disagree politely.",
  "ok": "Expressing agreement, acceptance, or acknowledgment.",
  "you're welcome": "A polite reply to thanks.",
  "peace": "A state of calm and harmony.",
  "please": "Used to make polite requests.",
  "sorry": "Used to express regret or apology.",
  "thank you": "Words expressing gratitude or appreciation.",
  "yes": "Used to agree or give consent.",
  "good bye": "Used when leaving or parting.",
  "happy": "Feeling joy, pleasure, or satisfaction.",
  "sad": "Feeling sorrow, unhappiness, or disappointment.",
  "angry": "Feeling strong displeasure or hostility.",
  "hate": "Intense dislike or strong aversion.",
  "good luck": "Wishing success or a positive outcome.",
  "live long and prosper": "A common Vulcan greeting expressing good wishes."
};

// ---- State -----------------------------------------------------------------------------------
let mode = "words";
let running = false;
let stream = null;
let hands = null;
let handsReady = null;          // promise
let labelsByMode = null;
const models = {};              // mode -> tf.LayersModel
const modelLoading = {};        // mode -> promise
let classifying = false;
let lastClassifyAt = 0;
let history = [];               // recent probability vectors for the current mode
let current = null;             // { label, confidence } currently shown
let lastHandAt = 0;

const cropCanvas = document.createElement("canvas");
cropCanvas.width = cropCanvas.height = IMG_SIZE;
const cropCtx = cropCanvas.getContext("2d");

// ---- UI helpers ------------------------------------------------------------------------------
function setStatus(text) { cameraStatus.textContent = text; }

function titleCase(text) { return text.replace(/\b\w/g, letter => letter.toUpperCase()); }

function showResult(label, confidence) {
  resultSymbol.textContent = mode === "letters" || mode === "numbers" ? label.toUpperCase() : "✓";
  resultLabel.textContent = mode === "words" ? titleCase(label) : label.toUpperCase();
  resultDescription.textContent = descriptions[label.toLowerCase()] ||
    (mode === "letters" ? "Letter recognized from the alphabet model." :
     mode === "numbers" ? "Number recognized from the numbers model." : "Sign recognized by the selected model.");
  const pct = Math.round(confidence * 1000) / 10;
  confidenceValue.textContent = `${pct}%`;
  confidenceBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

function showUnsure() {
  resultSymbol.textContent = "?";
  resultLabel.textContent = "Not sure yet";
  resultDescription.textContent = "Hold your sign steady and keep your whole hand in view.";
  confidenceValue.textContent = "—";
  confidenceBar.style.width = "0%";
}

function resetResult(message = "Your recognized sign will show here.") {
  resultSymbol.textContent = "?";
  resultLabel.textContent = "Waiting for a sign";
  resultDescription.textContent = message;
  confidenceValue.textContent = "—";
  confidenceBar.style.width = "0%";
}

// ---- Loading hand detector + models ----------------------------------------------------------
function loadHands() {
  if (handsReady) return handsReady;
  handsReady = (async () => {
    if (typeof Hands === "undefined") throw new Error("Hand detector script failed to load.");
    const detector = new Hands({ locateFile: file => BASE + "vendor/hands/" + file });
    detector.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,          // same as cvzone's default
      minDetectionConfidence: 0.5, // same as cvzone's default
      minTrackingConfidence: 0.5
    });
    detector.onResults(handleResults);
    await detector.initialize();
    hands = detector;
  })();
  handsReady.catch(() => { handsReady = null; });
  return handsReady;
}

async function loadLabels() {
  if (labelsByMode) return labelsByMode;
  const response = await fetch(BASE + "models/labels.json");
  if (!response.ok) throw new Error("Could not load sign labels.");
  labelsByMode = await response.json();
  return labelsByMode;
}

function loadModel(name) {
  if (models[name]) return Promise.resolve(models[name]);
  if (modelLoading[name]) return modelLoading[name];
  modelLoading[name] = (async () => {
    await loadLabels();
    const model = await tf.loadLayersModel(MODEL_URLS[name]);
    tf.tidy(() => model.predict(tf.zeros([1, MODEL_SIZE, MODEL_SIZE, 3])).dataSync()); // warm up
    models[name] = model;
    return model;
  })();
  modelLoading[name].catch(() => { delete modelLoading[name]; });
  return modelLoading[name];
}

// ---- Drawing the training-style skeleton (matches cvzone's findHands(draw=True)) -------------
function drawTrainingStyle(ctx, points, bbox) {
  const [x, y, w, h] = bbox;
  // connections: light grey, 2px (MediaPipe default)
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgb(224,224,224)";
  ctx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.moveTo(points[a][0], points[a][1]);
    ctx.lineTo(points[b][0], points[b][1]);
  }
  ctx.stroke();
  // landmarks: red dot with white ring (MediaPipe default)
  for (const [px, py] of points) {
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = "rgb(255,0,0)";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgb(255,255,255)";
    ctx.stroke();
  }
  // magenta rectangle cvzone draws around the hand
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgb(255,0,255)";
  ctx.strokeRect(x - OFFSET, y - OFFSET, w + 2 * OFFSET, h + 2 * OFFSET);
}

// ---- Cropping like datacollection.py ---------------------------------------------------------
function prepareHandImage(bbox) {
  const [x, y, w, h] = bbox;
  if (w <= 0 || h <= 0) return false;
  const fw = frameCanvas.width, fh = frameCanvas.height;
  const x1 = Math.max(0, x - OFFSET), y1 = Math.max(0, y - OFFSET);
  const x2 = Math.min(fw, x + w + OFFSET), y2 = Math.min(fh, y + h + OFFSET);
  if (x2 - x1 < 2 || y2 - y1 < 2) return false;

  cropCtx.fillStyle = "#fff";
  cropCtx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
  cropCtx.imageSmoothingEnabled = true;
  cropCtx.imageSmoothingQuality = "high";

  if (h / w > 1) {
    const newW = Math.max(1, Math.ceil((IMG_SIZE / h) * w));
    const gap = Math.ceil((IMG_SIZE - newW) / 2);
    cropCtx.drawImage(frameCanvas, x1, y1, x2 - x1, y2 - y1, gap, 0, newW, IMG_SIZE);
  } else {
    const newH = Math.max(1, Math.ceil((IMG_SIZE / w) * h));
    const gap = Math.ceil((IMG_SIZE - newH) / 2);
    cropCtx.drawImage(frameCanvas, x1, y1, x2 - x1, y2 - y1, 0, gap, IMG_SIZE, newH);
  }
  return true;
}

// ---- Overlay (bounding box + label), positioned over the object-fit: cover video -------------
function clearOverlay() {
  overlayCtx.setTransform(1, 0, 0, 1, 0, 0);
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function fitOverlay() {
  const box = overlay.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(box.width * dpr), h = Math.round(box.height * dpr);
  if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return box;
}

function drawBoundingBox(bbox) {
  const box = fitOverlay();
  overlayCtx.clearRect(0, 0, box.width, box.height);
  const sourceW = frameCanvas.width, sourceH = frameCanvas.height;
  if (!sourceW || !sourceH) return;

  const [x, y, w, h] = bbox;
  const scale = Math.max(box.width / sourceW, box.height / sourceH);   // object-fit: cover
  const offsetX = (box.width - sourceW * scale) / 2;
  const offsetY = (box.height - sourceH * scale) / 2;

  // Same padded box that cvzone draws (hand box + OFFSET on every side).
  const left = offsetX + (x - OFFSET) * scale;
  const top = offsetY + (y - OFFSET) * scale;
  const width = (w + 2 * OFFSET) * scale;
  const height = (h + 2 * OFFSET) * scale;

  const bad = current && OFFENSIVE_LABELS.has(current.label.toLowerCase());
  const color = bad ? "#ff4d4d" : "#35e28a";

  overlayCtx.lineWidth = 3;
  overlayCtx.strokeStyle = color;
  overlayCtx.shadowColor = "rgba(0,0,0,.35)";
  overlayCtx.shadowBlur = 5;
  overlayCtx.strokeRect(left, top, width, height);
  overlayCtx.shadowBlur = 0;

  const tag = current
    ? `${mode === "words" ? titleCase(current.label) : current.label.toUpperCase()}  ${Math.round(current.confidence * 100)}%`
    : "HAND DETECTED";
  overlayCtx.font = "bold 13px system-ui, sans-serif";
  const tagWidth = overlayCtx.measureText(tag).width + 18;
  const tagTop = top - 26 < 0 ? top + 4 : top - 26;
  overlayCtx.fillStyle = color;
  overlayCtx.fillRect(left - 1.5, tagTop, tagWidth, 24);
  overlayCtx.fillStyle = "#10251a";
  overlayCtx.fillText(tag, left + 8, tagTop + 16);
}

// ---- Per-frame results from MediaPipe --------------------------------------------------------
function handleResults(results) {
  if (!running) return;                                       // ignore results that arrive after Stop
  const landmarks = results.multiHandLandmarks && results.multiHandLandmarks[0];
  if (!landmarks) {
    clearOverlay();
    if (Date.now() - lastHandAt > 700) {
      history = [];
      current = null;
      if (running) setStatus("Show one hand inside the camera.");
    }
    return;
  }
  lastHandAt = Date.now();

  const fw = frameCanvas.width, fh = frameCanvas.height;
  // cvzone: px = int(lm.x * w)
  const points = landmarks.map(p => [Math.trunc(p.x * fw), Math.trunc(p.y * fh)]);
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const xmin = Math.min(...xs), ymin = Math.min(...ys);
  const bbox = [xmin, ymin, Math.max(...xs) - xmin, Math.max(...ys) - ymin];

  drawBoundingBox(bbox);

  const now = performance.now();
  const model = models[mode];
  if (!model) { setStatus("Loading the recognition model…"); return; }
  if (classifying || now - lastClassifyAt < CLASSIFY_EVERY_MS) return;

  // The model was trained on frames that already had the skeleton drawn on them.
  drawTrainingStyle(frameCtx, points, bbox);
  if (!prepareHandImage(bbox)) return;

  lastClassifyAt = now;
  classifying = true;
  const requestMode = mode;
  classify(model, requestMode).finally(() => { classifying = false; });
}

async function classify(model, requestMode) {
  let input, output;
  try {
    input = tf.tidy(() => {
      const pixels = tf.browser.fromPixels(cropCanvas);                         // RGB, like training
      const resized = tf.image.resizeBilinear(pixels.toFloat(), [MODEL_SIZE, MODEL_SIZE]);
      return resized.div(127.5).sub(1).expandDims(0);                           // Teachable Machine scaling
    });
    output = model.predict(input);
    const probs = Array.from(await output.data());
    if (!running || requestMode !== mode) return;                               // stopped / mode changed meanwhile

    history.push(probs);
    if (history.length > SMOOTHING_FRAMES) history.shift();
    const avg = probs.map((_, i) => history.reduce((sum, p) => sum + p[i], 0) / history.length);
    let best = 0;
    avg.forEach((value, i) => { if (value > avg[best]) best = i; });

    const labels = labelsByMode[requestMode];
    if (avg[best] >= MIN_CONFIDENCE && labels[best] !== undefined) {
      current = { label: labels[best], confidence: avg[best] };
      showResult(current.label, current.confidence);
      setStatus("Hand detected. Recognition is running.");
    } else {
      current = null;
      showUnsure();
      setStatus("Hand detected. Hold the sign steady.");
    }
  } catch (error) {
    console.error(error);
    setStatus("Recognition error: " + (error.message || error));
  } finally {
    if (input) input.dispose();
    if (output) output.dispose();
  }
}

// ---- Camera loop -----------------------------------------------------------------------------
async function processFrames() {
  while (running) {
    try {
      if (hands && video.readyState >= 2 && video.videoWidth) {
        const width = Math.min(video.videoWidth, FRAME_WIDTH);
        const height = Math.round(video.videoHeight * width / video.videoWidth);
        if (frameCanvas.width !== width || frameCanvas.height !== height) {
          frameCanvas.width = width;
          frameCanvas.height = height;
        }
        // Mirror the frame, exactly like cv2.flip(img, 1) in the original scripts.
        frameCtx.setTransform(-1, 0, 0, 1, width, 0);
        frameCtx.drawImage(video, 0, 0, width, height);
        frameCtx.setTransform(1, 0, 0, 1, 0, 0);
        await hands.send({ image: frameCanvas });
      }
    } catch (error) {
      console.error(error);
      setStatus("Hand detection error: " + (error.message || error));
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    // Wait for the next screen refresh (keeps the loop friendly to the browser).
    await new Promise(resolve => requestAnimationFrame(resolve));
  }
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("Camera access needs HTTPS or localhost, and a supported browser.");
    return;
  }
  startButton.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    video.style.display = "block";
    overlay.style.display = "block";
    placeholder.style.display = "none";
    cameraHint.style.display = "block";
    stopButton.disabled = false;
    liveDot.style.background = "#38a66b";
  } catch (error) {
    startButton.disabled = false;
    setStatus(error.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access in your browser."
      : "Could not start the camera. Check that it is connected and not being used by another app.");
    return;
  }

  running = true;
  setStatus("Camera is on. Loading hand detector and models (first time only)…");
  try {
    await Promise.all([loadHands(), loadModel(mode)]);
  } catch (error) {
    console.error(error);
    setStatus("Could not load the recognition files: " + (error.message || error));
    stopCamera(false);
    return;
  }
  if (!running) return;
  setStatus("Camera is on. Show a sign.");
  processFrames();
}

function stopCamera(updateStatus = true) {
  running = false;
  classifying = false;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  video.style.display = "none";
  overlay.style.display = "none";
  clearOverlay();
  placeholder.style.display = "flex";
  cameraHint.style.display = "none";
  startButton.disabled = false;
  stopButton.disabled = true;
  liveDot.style.background = "#b8c0ba";
  history = [];
  current = null;
  if (updateStatus) setStatus("Camera is off.");
  resetResult();
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", () => stopCamera());

modeButtons.forEach(button => {
  button.addEventListener("click", async () => {
    mode = button.dataset.mode;
    modeButtons.forEach(tab => {
      const active = tab === button;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    history = [];
    current = null;
    clearOverlay();
    resetResult(`Now recognizing ${mode}. Show a sign.`);
    if (running) setStatus(`Mode changed to ${mode}. Show a sign.`);
    try {
      if (!models[mode]) {
        if (running) setStatus(`Loading the ${mode} model…`);
        await loadModel(mode);
        if (running) setStatus(`Mode changed to ${mode}. Show a sign.`);
      }
    } catch (error) {
      console.error(error);
      setStatus("Could not load the model: " + (error.message || error));
    }
  });
});

window.addEventListener("resize", clearOverlay);
window.addEventListener("beforeunload", () => stopCamera(false));
