# Korean Handwriting Generation

Generate Korean (Hangul) handwriting **stroke by stroke** with a neural network,
learned from real pen-tablet recordings.

A GRU recurrent network with a Graves-style *sliding attention window* is
conditioned on the **jamo** (component letters) of each syllable and
autoregressively predicts pen motion — `(dx, dy, pen-lift)` at each step. Trained
on a single writer's hand, it learns to produce that writer's Hangul.

## Repository layout

| Directory | What it is |
|-----------|------------|
| [`handwriting-generation/`](handwriting-generation/) | The PyTorch model, training pipeline, and inference. A bundled trained model (`best_model.emb8.pt`) is included so you can generate right away. |
| [`collection-app/`](collection-app/) | A small web app (React + Node + SQLite) for collecting handwriting from a Neo smartpen — workers are assigned sentences and their strokes are recorded and exported for training. Also stores feedback submitted from the demo site, with an admin page that can promote failed sentences into the writers' assignment pool. |
| [`demo-app/`](demo-app/) | The public demo at **[hangul.ink](https://hangul.ink)** — the model exported to ONNX runs entirely in the browser, streaming strokes into a live pen animation. Deployed on Cloudflare (static assets only). |

Design notes and the full progression of experiments live in
[`handwriting-generation/CLAUDE.md`](handwriting-generation/CLAUDE.md).

## How it works

A few ideas do most of the heavy lifting:

- **Jamo decomposition.** Rather than treat each of ~11,000 Hangul syllables as
  its own class, every syllable is conditioned on its three component jamo
  (leading consonant, vowel, trailing consonant) — 68 primitives that recombine.
  Far more data-efficient, and space/punctuation get their own tokens.
- **Sliding attention window** (Graves, 2013). A Gaussian window slides
  monotonically along the character sequence, telling the network *which*
  character it is drawing at each pen step, and terminating when it runs off the
  end of the text.
- **Mixture density output.** The next pen offset is a mixture of bivariate
  Gaussians, so at branch points (e.g. closing a loop) the model *samples* one
  mode instead of averaging them into mush.
- **Absolute-position inputs.** Cumulative pen position is fed alongside the
  deltas, so the model knows where it is along the line — this is what fixed the
  early "compressed layout" failure.
- **Full skip connections** (input→every layer, every layer→output). The output
  skip turned out to be necessary for the end-of-stroke head to learn at all.

## Quick start — generate handwriting

The repo ships with a trained model, so no training or data is needed to try it.

```bash
cd handwriting-generation
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# generate strokes for a string (window handles termination on multi-char text)
NUM_LAYERS=3 EMBEDDING_SIZE=8 python inference.py \
  --model best_model.emb8.pt --character "안녕하세요" --output generated.json

# render to SVG and open it in a browser
python render.py generated.json out.svg
```

`NUM_LAYERS` / `EMBEDDING_SIZE` must match how the model was trained (the bundled
model is a 3-layer, embedding-size-8 network — the current best).

## Training

Training reads the exported recordings from `../data/export/` (collected via the
collection app; not committed). With data in place:

```bash
cd handwriting-generation
source venv/bin/activate
EXPORT=1 NUM_LAYERS=3 EMBEDDING_SIZE=8 python train.py
```

Knobs are environment variables: `NUM_LAYERS`, `HIDDEN_SIZE`, `EMBEDDING_SIZE`,
`BATCH_SIZE`, `LR`, `NUM_EPOCHS`, and `RESUME` / `INIT_WEIGHTS` for continuing a
run. See `CLAUDE.md` for details.

## Collection app

```bash
cd collection-app
npm install
npm run dev        # see collection-app/README.md for admin/worker setup
```

## Status

Trained on ~800 lines from a single writer, the output is **nearly legible** —
line layout matches the human hand and most individual jamo are correct. The main
lever for further quality is simply more collected data.

## License

_TODO: add a license before publishing._ (This project was originally scaffolded
from a pen-SDK sample; all sample code has been removed, but pick a license that
suits how you want the work used.)
