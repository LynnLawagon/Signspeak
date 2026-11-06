const video = document.getElementById("camera");
const canvas = document.getElementById("capture");
const context = canvas.getContext("2d");
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

const descriptions = {
  "hello": "A friendly greeting.",
  "thank you": "A way to show gratitude.",
  "sorry": "Used to express an apology.",
  "please": "A polite word used when asking.",
  "you're welcome": "A polite reply to thanks.",
  "good bye": "A greeting used when leaving.",
  "i love you": "An expression of affection.",
  "yes": "Used to agree or give consent.",
  "no": "Used to refuse or disagree."
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

async function recognizeFrame() {
  if (!stream || busy || video.readyState < 2) return;
  busy = true;
  try {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, image: canvas.toDataURL("image/jpeg", 0.7) })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    if (data.detected) {
      showResult(data.label, data.confidence);
      cameraStatus.textContent = "Hand detected. Recognition is running.";
    } else {
      cameraStatus.textContent = data.message || "Show your hand to the camera.";
    }
  } catch (error) {
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
    placeholder.style.display = "none";
    cameraHint.style.display = "block";
    startButton.disabled = true;
    stopButton.disabled = false;
    document.querySelector(".live-indicator i").style.background = "#38a66b";
    cameraStatus.textContent = "Camera is on. Show a sign.";
    timer = window.setInterval(recognizeFrame, 700);
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
    resetResult(`Now recognizing ${mode}. Show a sign.`);
    if (stream) cameraStatus.textContent = `Mode changed to ${mode}. Show a sign.`;
  });
});

window.addEventListener("beforeunload", stopCamera);
