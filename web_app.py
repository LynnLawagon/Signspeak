"""
SignSpeak web server.

Hand detection (MediaPipe Hands) and sign classification (the Teachable Machine models,
converted to TensorFlow.js) now run inside the visitor's browser, so this server only has to
serve the page and its static files. That keeps it tiny: no TensorFlow, OpenCV or MediaPipe
is needed on Render, it starts in a second, and it fits comfortably on a free instance.

The original Keras models (*/keras_model.h5) and the desktop scripts (test.py,
datacollection.py) are untouched. See README.md for how the browser models were produced.
"""
import mimetypes
import os

from flask import Flask, jsonify, render_template

# Make sure the WebAssembly / module files used by the hand detector get the right type.
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/octet-stream", ".tflite")
mimetypes.add_type("application/octet-stream", ".binarypb")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_NAMES = ("words", "letters", "numbers")

app = Flask(__name__)
# Vendor files and models never change between deploys; app.js / styles.css are versioned
# with ?v=<mtime> in the template, so a long cache is safe and makes repeat visits fast.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 60 * 60 * 24 * 7


def asset_version():
    newest = 0
    for name in ("app.js", "styles.css"):
        try:
            newest = max(newest, int(os.path.getmtime(os.path.join(STATIC_DIR, name))))
        except OSError:
            pass
    return newest


@app.get("/")
def index():
    response = app.make_response(render_template("index.html", v=asset_version()))
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/health")
def health():
    available = [
        name for name in MODEL_NAMES
        if os.path.isfile(os.path.join(STATIC_DIR, "models", name, "model.json"))
    ]
    return jsonify({
        "status": "ok" if len(available) == len(MODEL_NAMES) else "missing-models",
        "models": available,
        "model_count": len(available),
        "inference": "browser"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
