# SignSpeak — Sign Language Recognition

A browser-based website for the existing Python + Teachable Machine sign-recognition project.

## What is included

- **Alphabet model:** A–Z (26 classes)
- **Number model:** 1–5 (5 classes)
- **Common words model:** 20 trained classes
- **Frontend:** HTML, CSS, and JavaScript
- **Backend:** Python, Flask, OpenCV, cvzone, and the original Teachable Machine `.h5` models

The trained model files and the original desktop scripts (`test.py` and `datacollection.py`) are kept in the project. The website uses `web_app.py`; it does not replace or retrain the models.

## Run locally (Python 3.10)

TensorFlow 2.10 is used by the original project and requires a compatible Python environment. Use Python 3.10.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python web_app.py
```

Open http://127.0.0.1:5000 in your browser. Allow camera access.

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python web_app.py
```

## Deploy to Render

1. Push this project folder to a GitHub repository.
2. In Render, choose **New + → Web Service** and connect the repository.
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn --workers 1 --threads 2 --timeout 300 --graceful-timeout 30 web_app:app`
4. Deploy, then open the HTTPS URL and allow camera access.

The `Procfile` is included for platforms that use it. The service uses one worker because the hand detector and model objects are loaded once and shared by requests. Two request threads are protected by a lock during inference. The 300-second timeout allows slower CPU-only startup/inference; if the host kills the process for memory use, use a larger instance.

## Important deployment notes

- Camera access in browsers requires HTTPS (or localhost during local development).
- The Python backend receives camera frames and runs hand detection + classification. This means the server needs enough memory and CPU for TensorFlow and MediaPipe. A small free hosting instance may run slowly or may not have enough memory; use a plan with adequate RAM/CPU if deployment fails or predictions are delayed.
- Do not deploy this backend as a static-only site (such as a plain Netlify static site). The Python API and model files must run on a Python server.
- The project recognizes individual signs, not continuous signed sentences. Predictions can be wrong; lighting, camera angle, hand position, and training data affect results.
- The model labels include an offensive gesture. It is retained because it is part of the existing trained model, not added by the website.

## Project structure

```text
Sign-language/
├── allLetters/
│   ├── keras_model.h5
│   └── labels.txt
├── converted_keras/
│   ├── keras_model.h5
│   └── labels.txt
├── numbers_converted_keras/
│   ├── keras_model.h5
│   └── labels.txt
├── templates/
│   └── index.html
├── static/
│   ├── app.js
│   └── styles.css
├── web_app.py
├── test.py                 # Original desktop recognition program
├── datacollection.py       # Original data collection program
├── requirements.txt
├── .python-version
└── Procfile
```

## API check

After starting the server, visit `/health`. It should return `status: ok` and list all three model modes.
