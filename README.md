# SignSpeak — Sign Language Recognition

A browser-based website for the existing Python + Teachable Machine sign-recognition project.

## What is included

- **Alphabet model:** A–Z (26 classes)
- **Number model:** 1–5 (5 classes)
- **Common words model:** 20 trained classes
- **Frontend:** HTML, CSS, JavaScript, TensorFlow.js and MediaPipe Hands (all self-hosted in `static/vendor`)
- **Backend:** a tiny Flask server that only serves the page and files

The original Keras models (`*/keras_model.h5`) and the desktop scripts (`test.py`, `datacollection.py`) are kept
unchanged. The website uses copies of the same models converted to TensorFlow.js (`static/models/`).

## How it works

Everything heavy now happens in the visitor's browser:

1. **MediaPipe Hands** (the same hand model cvzone uses) tracks the hand and draws the live bounding box.
2. The hand is cropped exactly like `datacollection.py` / `test.py` do it: mirrored frame, hand skeleton and pink box
   drawn on the frame, 20 px offset, hand centred on a 300×300 white square. **This matters** – the models were trained
   on images that had the skeleton drawn on them, so clean crops are recognised poorly.
3. **TensorFlow.js** runs the converted Teachable Machine model for the selected category.
4. The last 5 predictions are averaged so the label does not flicker.

Because the server no longer runs TensorFlow, OpenCV or MediaPipe, it starts in about a second, needs very little
memory, and works on Render's free plan.

## Run locally

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python web_app.py
```

Open http://127.0.0.1:5000 and allow camera access. Any Python 3.9+ works for the website.

The original desktop scripts still need Python 3.10 and `pip install -r requirements-desktop.txt`.

## Deploy to Render

1. Push this project folder to a GitHub repository.
2. In Render, choose **New + → Web Service** and connect the repository.
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn --workers 1 --threads 4 --timeout 120 web_app:app`
4. Deploy, open the HTTPS URL and allow camera access.

The first visit downloads about 15–20 MB (hand detector + models). They are cached afterwards.
Camera access requires HTTPS (Render provides it) or localhost.

Check `/health` – it should report `status: ok` and list `words`, `letters` and `numbers`.

## Re-exporting the models (only if you retrain)

If you retrain in Teachable Machine and export a new Keras `.h5`, convert it for the browser with
`tensorflowjs` (e.g. `tensorflowjs_converter --input_format keras keras_model.h5 static/models/<mode>/`) and update
`static/models/labels.json`. Alternatively, export the **TensorFlow.js** option from Teachable Machine directly.

## Notes

- The project recognizes individual signs, not continuous signed sentences. Lighting, camera angle, hand position and
  training data all affect accuracy.
- The word model includes an offensive gesture. It is retained because it is part of the existing trained model; the
  bounding box turns red when it is detected.

## Project structure

```text
Sign-language/
├── allLetters/ converted_keras/ numbers_converted_keras/   # original Keras models (desktop scripts)
├── static/
│   ├── app.js            # camera, hand tracking, preprocessing, prediction, bounding box
│   ├── styles.css
│   ├── models/           # TensorFlow.js models + labels.json (used by the website)
│   └── vendor/           # tf.min.js and MediaPipe Hands (self-hosted, no CDN)
├── templates/index.html
├── web_app.py            # Flask: serves the page + /health
├── test.py, datacollection.py   # original desktop programs
├── requirements.txt            # website (Flask + gunicorn)
├── requirements-desktop.txt    # only for the desktop scripts
├── RENDER-START-COMMAND.txt
├── .python-version
└── Procfile
```
