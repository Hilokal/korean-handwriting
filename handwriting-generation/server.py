"""Flask server for batch handwriting inference.

Wraps the existing inference code so the React `/inference` page can run the
PyTorch model. Reuses load_model / generate_sequence / sequence_to_json /
get_device from inference.py and the tokenizer for input validation -- no model
logic is duplicated here.

Run:  venv/bin/python server.py   (listens on http://localhost:5001)
"""

import os
import random
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from inference import (
    generate_sequence,
    get_device,
    load_model,
    sequence_to_json,
)
from handwriting_dataset import modal_stroke_counts
from tokenizer import decompose_hangul_syllable

HERE = Path(__file__).resolve().parent
PORT = 5001

app = Flask(__name__)
CORS(app)

device = get_device()

# Expected stroke count per character, used to terminate generation.
STROKE_COUNTS = modal_stroke_counts()

# Cache loaded models by checkpoint path so repeated requests are fast.
_model_cache = {}


def get_cached_model(model_name: str):
    """Load a checkpoint (relative to this dir), caching by resolved path."""
    path = (HERE / model_name).resolve()
    # Keep checkpoints inside this directory; reject path traversal.
    if HERE not in path.parents or not path.name.endswith(".pt"):
        raise ValueError(f"Invalid model path: {model_name}")
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_name}")

    key = str(path)
    if key not in _model_cache:
        _model_cache[key] = load_model(str(path), device)
    return _model_cache[key]


def list_models():
    return sorted(f.name for f in HERE.glob("*.pt"))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": str(device)})


@app.get("/models")
def models():
    return jsonify({"models": list_models()})


@app.post("/generate")
def generate():
    body = request.get_json(silent=True) or {}

    character = body.get("character", "")
    count = int(body.get("count", 1))
    temperature = float(body.get("temperature", 1.0))
    max_len = int(body.get("maxLen", 500))
    bias = float(body.get("bias", 0.0))
    model_name = body.get("model", "best_model.pt")
    base_seed = body.get("seed", None)
    # Stop after the character's expected stroke count (override with numStrokes).
    num_strokes = body.get("numStrokes", None)

    # --- validation ---
    if not isinstance(character, str) or len(character) != 1:
        return jsonify({"error": "character must be a single character"}), 400
    try:
        decompose_hangul_syllable(character)  # raises on invalid syllable
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if count < 1 or count > 64:
        return jsonify({"error": "count must be between 1 and 64"}), 400

    # A random base when none is given; each item gets a distinct, reported seed.
    if base_seed is None or base_seed == "":
        base_seed = random.randint(0, 2**31 - 1)
    else:
        base_seed = int(base_seed)

    try:
        model = get_cached_model(model_name)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400

    if num_strokes is None:
        num_strokes = STROKE_COUNTS.get(character)
    else:
        num_strokes = int(num_strokes)

    results = []
    for i in range(count):
        seed = base_seed + i
        sequence = generate_sequence(
            model,
            character,
            device,
            max_len=max_len,
            temperature=temperature,
            seed=seed,
            bias=bias,
            num_strokes=num_strokes,
        )
        data = sequence_to_json(sequence)
        results.append(
            {
                "seed": seed,
                "numPoints": len(data["dots"]),
                "dots": data["dots"],
            }
        )

    return jsonify(
        {
            "device": str(device),
            "model": model_name,
            "character": character,
            "results": results,
        }
    )


if __name__ == "__main__":
    print(f"Inference server on http://localhost:{PORT} (device: {device})")
    print(f"Available models: {', '.join(list_models()) or '(none)'}")
    app.run(host="127.0.0.1", port=PORT, debug=False)
