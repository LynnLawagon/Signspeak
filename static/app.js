const video = document.getElementById("camera");
const canvas = document.getElementById("capture");
const context = canvas.getContext("2d");
const overlay = document.getElementById("handOverlay");
const overlayContext = overlay.getContext("2d");
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

let stream = null;
let mode = "words";
let timer = null;
let busy = false;
let lastRequestAt = 0;

const descriptions = {
  "hello": "A friendly greeting.",
  "thank you": "A way to show gratitude.",
  "sorry": "Used to express an apology.",
  "please": "A polite word used when asking.",
  "you're welcome": "A polite reply to thanks.",
  "good bye": "A greeting used when leaving."
};

function showResult(label, confidence) {
  resultSymbol.textContent = mode === "letters" || mode === "numbers" ? label.toUpperCase() : "✓";
  resultLabel.textContent = label.replace(/\b\w/g, letter => letter.toUpperCase());
  resultDescription.textContent = descriptions[label.toLowerCase()] || "Sign recognized by the selected model.";
  confidenceValue.textContent = `${confidence}%`;
  confidenceBar.style.width = `${Math.max(0, Math.min(100, confidence))}%`;
}

function resetResult(message = "Your recognized sign will show here.") {
  resultSymbol.textContent = "?";
  resultLabel.textContent = "Waiting for a sign";
  resultDescription.textContent = message;
  confidenceValue.textContent = "—";
  confidenceBar.style.width = "0%";
}

function clearOverlay() {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
}

function drawBoundingBox(data) {
  clearOverlay();
  if (!data.detected || !data.bbox || !video.videoWidth) return;

  const [x, y, w, h] = data.bbox;
  const sourceW = data.frame_width || canvas.width;
  const sourceH = data.frame_height || canvas.height;

  // Match the video's object-fit: cover layout.
  const box = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(box.width * dpr);
  overlay.height = Math.round(box.height * dpr);
  overlayContext.setTransform(dpr, 0, 0, dpr, 0, 0);

  const scale = Math.max(box.width / sourceW, box.height / sourceH);
  const renderedW = sourceW * scale;
  const renderedH = sourceH * scale;
  const offsetX = (box.width - renderedW) / 2;
  const offsetY = (box.height - renderedH) / 2;

  const left = offsetX + x * scale;
  const top = offsetY + y * scale;
  const width = w * scale;
  const height = h * scale;

  overlayContext.lineWidth = 3;
  overlayContext.strokeStyle = "#35e28a";
  overlayContext.shadowColor = "rgba(0,0,0,.35)";
  overlayContext.shadowBlur = 5;
  overlayContext.strokeRect(left, top, width, height);
  overlayContext.shadowBlur = 0;

  overlayContext.fillStyle = "#35e28a";
  overlayContext.font = "bold 12px system-ui";
  const tag = "HAND DETECTED";
  const tagWidth = overlayContext.measureText(tag).width + 16;
  const tagTop = Math.max(0, top - 25);
  overlayContext.fillRect(left, tagTop, tagWidth, 22);
  overlayContext.fillStyle = "#10251a";
  overlayContext.fillText(tag, left + 8, tagTop + 15);
}

async function recognizeFrame() {
  if (!stream || busy || video.readyState < 2 || !video.videoWidth) return;
  busy = true;
  try {
    // Send a smaller image to reduce CPU and memory use on the hosted server.
    const targetWidth = Math.min(video.videoWidth, 480);
    const targetHeight = Math.round(video.videoHeight * targetWidth / video.videoWidth);
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    context.drawImage(video, 0, 0, targetWidth, targetHeight);

    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, image: canvas.toDataURL("image/jpeg", 0.65) })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Recognition server error (${response.status}).`);

    if (data.detected) {
      drawBoundingBox(data);
      showResult(data.label, data.confidence);
      cameraStatus.textContent = "Hand detected. Recognition is running.";
    } else {
      clearOverlay();
      cameraStatus.textContent = data.message || "Show your hand to the camera.";
    }
  } catch (error) {
    clearOverlay();
    cameraStatus.textContent = error.message || "Could not connect to the recognition server.";
  } finally {
    busy = false;
  }
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraStatus.textContent = "Camera access needs HTTPS or localhost, and a supported browser.";
    return;
  }
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
    startButton.disabled = true;
    stopButton.disabled = false;
    document.querySelector(".live-indicator i").style.background = "#38a66b";
    cameraStatus.textContent = "Camera is on. Show a sign.";
    // Avoid overlapping calls: each request completes before the next begins.
    timer = window.setInterval(() => {
      const now = Date.now();
      if (!busy && now - lastRequestAt >= 1200) {
        lastRequestAt = now;
        recognizeFrame();
      }
    }, 250);
  } catch (error) {
    cameraStatus.textContent = error.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access in your browser."
      : "Could not start the camera. Check that it is connected and not being used by another app.";
  }
}

function stopCamera() {
  if (timer) window.clearInterval(timer);
  timer = null;
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
  document.querySelector(".live-indicator i").style.background = "#b8c0ba";
  cameraStatus.textContent = "Camera is off.";
  resetResult();
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);

modeButtons.forEach(button => {
  button.addEventListener("click", () => {
    mode = button.dataset.mode;
    modeButtons.forEach(tab => {
      const active = tab === button;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    clearOverlay();
    resetResult(`Now recognizing ${mode}. Show a sign.`);
    if (stream) cameraStatus.textContent = `Mode changed to ${mode}. Show a sign.`;
  });
});

window.addEventListener("resize", clearOverlay);
window.addEventListener("beforeunload", stopCamera);
