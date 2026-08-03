# handwriting-generation

A small PyTorch experiment that learns to generate Korean (Hangul) handwriting
stroke-by-stroke from pen-tablet recordings. A GRU RNN with a **Graves-style
sliding attention window** is conditioned on a sequence of characters (each a
jamo triple, plus space/punctuation tokens) and autoregressively predicts the
next pen point `(dx, dy, penState)`.

## Environment

- Python with PyTorch. Create a venv and install deps:
  `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
- Device is auto-selected in this order: CUDA → Apple MPS → CPU.
- `venv/`, checkpoints, and `../data/` are gitignored; the one bundled model
  `best_model.emb8.pt` is committed so inference works out of the box.

## Files

| File | Role |
|------|------|
| `tokenizer.py` | `tokenize(text)` → `(U, 4)` Long tensor, one row per unit: `[leading, vowel, trailing, symbol]`. Hangul syllables decompose into the three jamo (symbol `0`); space/`.`/`,`/`!`/`?` use jamo `0` and a symbol id ≥ 1; unknown chars map to the unknown-symbol id (never raises). `decompose_hangul_syllable` is the standard Unicode algorithm (`trailingIndex == 0` = no 받침). |
| `handwriting_dataset.py` | `HandwritingDataset` loads the single-char folders `../data/<character>/*.json` (the `characters` list). `ExportDataset` loads the production export `../data/export/recordings/**/*.json` — real multi-character lines, transcript in `metadata.text`. Both yield `{strokes: (N,3), tokens: (U,4)}`. `transform()` (a staticmethod) converts dots → deltas. `split_by_text()` holds out whole sentences for the val set. |
| `synthetic_dataset.py` | `concat_samples()` stitches single-char recordings into synthetic multi-char "lines" (left-to-right, gap ∝ glyph width); `SyntheticMultiCharDataset` builds a set with mixed U. Exercises the attention window before real line data is available. |
| `model.py` | `HandwritingRNN`: jamo + symbol embeddings → `GRUWithSlidingAttention` (custom `GRUCell` loop with the Graves window) + an **MDN** head over `(dx, dy)` and a **binary** end-of-stroke head. Holds `mdn_params()` and `generate()` for autoregressive sampling. |
| `train.py` | Training loop + `mdn_loss` (bivariate-Gaussian NLL). Dataset chosen by env var (see Running). Early stopping; saves the best checkpoint to `best_model.pt`. |
| `inference.py` | CLI to load a checkpoint and generate strokes for a character/string, writing `generated.json`. |
| `render.py` | Renders `generated.json` (delta-encoded output) to an SVG file. |
| `server.py` | Optional Flask API for batched generation (see Batch inference server). |
| `download_export.sh` | Downloads all recordings from the production collection app into `../data/export/` via the admin export endpoint. Secure password prompt; re-run to refresh. |
| `fetch_reference_texts.py` | Fetches clean Hangul sentences from Korean Wikipedia into `../data/reference-texts/` (per-page `.txt` + `all.jsonl`, gitignored). Filters to pure-Hangul sentences and reports jamo/syllable coverage. Supports `--featured`/`--good`/`--category` bulk fetching, `--resume`, and authenticated fetching via `WIKI_USERNAME`/`WIKI_BOT_PASSWORD` env vars (bot password; anonymous requests get heavily rate-limited). |
| `best_model.emb8.pt` | The bundled trained model (3-layer, hidden 128, embedding_size 8). Load with `NUM_LAYERS=3 EMBEDDING_SIZE=8`. Training writes new checkpoints to `best_model.pt` (gitignored). |

## Data

- Line data is collected by the `collection-app/` in this repo (a Neo smartpen
  web app where workers write assigned sentences). Deploy it, then pull the
  exported recordings into `../data/export/` with `download_export.sh`. (Early
  single-character experiments used per-character folders `../data/<char>/*.json`,
  still loadable via `HandwritingDataset`.)
- Single-char recordings live in `../data/<character>/handwriting-segment-*.json`;
  exported lines in `../data/export/recordings/`. Each file has
  `{startTime, endTime, dotCount, dots, metadata}`; each dot has
  `x, y, dotType, timeStamp, ...`. Exported files add `metadata.text` (the line
  actually written), so the transcript travels with the file.
- `HandwritingDataset.transform()` (shared by both datasets) does the conversion:
  - Keeps only dots with `dotType == 1` (pen-move points); other dotTypes act as
    stroke boundaries.
  - Converts absolute coords to **deltas** `(dx, dy)`.
  - Drops glitch points whose `|dx|` or `|dy|` exceeds `max_delta=10.0` (delta is
    measured from the last *kept* point). This catches large jumps but *misses* a
    stray point sitting far off the line yet reached by a sub-`max_delta` delta —
    it renders as a long thin spike and stretches the whole recording. So a prior
    pass drops any pen-move point more than `y_outlier=3.0` from the recording's
    median line height (real lines are ~2.5 tall, p99 3.3). It repairs the *point*
    and keeps the line, rather than discarding the whole recording.
  - `penState` (**binary, Graves-style "trailing edge" anchoring**): `0` =
    mid-stroke, `1` = **last** point of a stroke (pen lifts *after* it, including
    the final stroke). A stroke-id pass derives this *after* filtering so the flag
    survives dropped glitch points. There is no end-of-sequence class; generation
    terminates by stroke count or by the window sliding past the last character
    (see Pen state). **Why trailing-edge:** `penState` is also a model *input*, so
    flagging the lift on the stroke's last point gives the recurrence a
    one-step-ahead "a stroke just ended" cue, letting it condition the upcoming
    between-stroke jump on it. (Originally the flag was on the *first* point of the
    next stroke; that gave no advance warning and produced slash-through-the-glyph
    jumps.)

## Token scheme

`tokenize(text)` returns a `(U, 4)` Long tensor, one row per unit:
`[leading, vowel, trailing, symbol]`.

- **Hangul syllable:** the three jamo indices, `symbol = 0`. Counts are
  `LeadingCount=19`, `VowelCount=21`, `TrailingCount=28` (27 trailing consonants +
  1 "none" at index 0).
- **Space / punctuation:** jamo slots `0`, `symbol ≥ 1` (`SYMBOLS = [" ", ".",
  ",", "!", "?"]`, plus an unknown-symbol id for anything else). These get their
  own conditioning token so their strokes attend to a real window position instead
  of bleeding into a neighbouring jamo's embedding. (Graves treated space and
  punctuation as characters too.)

In `model.py`, Hangul units are conditioned on `concat(leading, vowel, trailing
embeddings)` (9-dim); symbol units on a `symbol_embeddings` table emitting the
same 9-dim vector. So the per-character conditioning vector `c_u` is 9-dim
regardless of type, the window vector `w_t` (a φ-weighted sum of `c_u`) is 9-dim,
and the GRU input width stays 12 (`3` pen dims + `9`).

## Model architecture

- **Embeddings:** three `nn.Embedding`s (one per jamo slot) plus one
  `symbol_embeddings`. `embedding_size=3` per jamo; the symbol table emits `3*3=9`.
- **`GRUWithSlidingAttention`** — a manual `GRUCell` loop (not `nn.GRU`) so the
  attention window can be computed mid-sequence and fed back. Per timestep:
  - Layer 0's input is `cat([x_t, w_{t-1}])` — the pen input plus the *previous*
    step's window vector (the only path by which character info enters the GRU).
  - The window is computed from the **first** hidden layer `h¹_t` (Graves
    eqs. 46-51): `α = exp(·)`, `β = exp(·)`, `κ_t = κ_{t-1} + exp(·)` (so κ is
    monotonic → the window only slides forward), `φ(t,u) = Σ_k α_k exp(-β_k
    (κ_k-u)²)`, `w_t = Σ_u φ(t,u)·c_u`. `φ` is masked to real (non-padding)
    characters. `kappa_head` bias is initialized to `-4` so the window advances
    ~1 character per ~55 points (matching the ~58 points/char measured on
    single-char data); without it `exp(0)=1` slides the window off the string in
    ~3 steps.
  - **Skip connections (Graves' full synthesis connectivity):** the raw input
    `x_t` feeds *every* layer (input skip); the window feeds every layer — layer 0
    gets `w_{t-1}`, higher layers get the current `w_t` (window skip); and the
    heads read *every* layer's hidden state concatenated, not just the top (output
    skip). So a higher layer's input is `cat([x_t, h_below, w_t])`. Dropout is on
    the vertical (layer-to-layer) path only, so the recurrent state stays clean.
    Requires `num_layers ≥ 2`.
  - Returns `(output, state, phi)` where `output` is `(B, S, hidden*num_layers)`
    (all layers concatenated) and `state = (hidden, w, kappa)` is threaded across
    calls — `generate()` steps one point at a time, so κ and w must persist or the
    window resets to the start.
  - The output skip matters concretely: the **end-of-stroke signal lives in
    layer 1, not the top layer**. Without it the pen head (reading only the top,
    MDN-specialized layer) collapsed to predicting the class prior and its loss
    never moved. See the progress log.
  - Non-topological differences from Graves that remain: GRU vs his LSTM, 2 layers
    vs 3, 128 vs 400 cells, and jamo-decomposition conditioning vs a one-hot
    character alphabet.
- **Output heads** read all layers' hidden states (`hidden_size * num_layers`):
  - `mdn_head` — an **MDN**: `K=20` *bivariate* Gaussians over the next offset
    `(dx, dy)`. Emits `6*K` numbers/step: `[π, μx, μy, σx, σy, ρ]` per component.
    `mdn_params()` splits the raw output and applies activations (softmax π,
    exp σ, tanh ρ).
  - `pen_head` (1) — a **binary** end-of-stroke logit (sigmoid).
- The output factorizes: `p(dx,dy,pen | h) = MDN(dx,dy | h) · Bernoulli(pen | h)`.
- **Why an MDN, not MSE?** MSE predicts the conditional *mean* offset, which
  averages the multimodal pen motion at branch points (e.g. the top of a circle)
  into a drift that never closes — the ㅇ came out as an open spiral. The MDN
  models the full distribution and *samples* one mode at generation, so loops
  close. (Graves 2013 formulation. Graves used delta inputs only; this model also
  appends standardized absolute position -- see progress log #6 / `abs_pos_dim`.)

### Pen state (anchoring)

- `pen_head` is a **binary** end-of-stroke head (sigmoid), factorized from the
  offset (read independently from the same hidden state), like Graves'
  end-of-stroke Bernoulli. Termination is *not* a learned end-of-sequence class:
  at generation it stops after a fixed stroke count (`modal_stroke_counts()`,
  per single character) or — for multi-character strings — when the window slides
  past the last character.
- The lift flag is **trailing-edge** (`1` = stroke's last point) so it precedes
  and conditions the next jump. This representation change (no new inputs) fixed
  the "vowel slashes through the ㅇ" jumps and gave proper ㅇ-left/ㅏ-right
  composition.
- **Rendering convention follows from this:** a point starts a new stroke (`M`)
  when the *previous* point was end-of-stroke (`penState == 1`), not the current
  one. `render.py` and the collection app's `strokeSvg.ts` both apply this.

## Training loss

- XY: `mdn_loss` — negative log-likelihood of the true offset under the mixture,
  computed in log-space via `logsumexp` for stability (see `train.py`).
- Pen: masked `BCEWithLogitsLoss` over the binary end-of-stroke flag, with
  `pos_weight=10` to upweight the rare positive (~1 per stroke). Masked to real
  (non-padding) steps.
- **The NLL can (and should) go negative** — a continuous density can exceed 1,
  so its `-log` is negative. More negative = more confident + accurate. Watch the
  *trend*, not the sign. Do **not** compare its magnitude to the old MSE loss
  (different units); compare the *rendered* output instead.
- Stability guards: `clamp(1 - ρ², min=1e-6)` in the loss and σ initialized to ~1
  via the `mdn_head` bias. If loss runs to huge negatives with NaNs, that's σ
  collapsing — add a σ floor.

## Running

Train — the dataset is chosen by an env var (writes `best_model.pt`):

```bash
EXPORT=1 python train.py      # real multi-character lines (../data/export/), held out by text
SYNTHETIC=1 python train.py   # stitched multi-char samples (attention-window smoke test)
python train.py               # original single-character folders (../data/<char>/)
```

Pull the latest line data from production first:

```bash
bash download_export.sh       # prompts for the admin password (never echoed)
```

Generate (writes `generated.json`):

```bash
python inference.py --character 안 --temperature 1.0 --bias 0 --output generated.json
# --character accepts a multi-character string; omit --num-strokes to let the
#   window terminate (single characters fall back to their modal stroke count).
# optional: --model <ckpt> --max-len 500 --seed <n> --seed-index <i> --seed-length 20
```

**Two sampling knobs** (they act on different heads):
- `--temperature` → the end-of-stroke sigmoid (categorical): *when* to lift/end
  strokes. `0` = greedy/deterministic.
- `--bias` → the MDN offset head (continuous mixture): *where* the pen goes.
  Higher = sharper π + smaller σ = cleaner, less varied strokes; `0` = unbiased
  (sample the model's true distribution). Effectively an *inverse* temperature
  for the offsets (`bias ≈ 1/T`).

## Viewing output

`generated.json` (delta-encoded model output) renders to SVG with `render.py`:

```bash
python render.py generated.json out.svg   # then open out.svg in a browser
```

It applies the same stroke-break convention as training: a point starts a new
stroke (SVG `M`) when the *previous* point was end-of-stroke (`penState == 1`).

## Batch inference server (optional)

`server.py` is a small Flask API that loads a model once and serves batched
generation for eyeballing many samples. It reuses `inference.py`'s
`load_model` / `generate_sequence` / `sequence_to_json` — no model logic
duplicated. Endpoints: `POST /generate`, `GET /models`, `GET /health`; caches
models per checkpoint path; runs on **port 5001** (5000 is taken by macOS
AirPlay). Deps in `requirements.txt`.

```bash
pip install -r requirements.txt
python server.py   # http://localhost:5001
```

Seeding: `generate_sequence(..., seed=)` (`torch.manual_seed`) is shared by the
server and the `--seed` CLI flag; the `POST /generate` body also accepts `bias`
and `temperature`. (The original browser front-end for this lived in the pen-SDK
sample app it was scaffolded from and is not part of this repo.)

## Gotchas

- **hidden_size / num_layers must match at load:** the checkpoint stores no
  architecture config, so a checkpoint must be loaded with the same `hidden_size`
  and `num_layers` it was trained with. Both `train.py` and `inference.py` read
  `HIDDEN_SIZE` / `NUM_LAYERS` env vars (defaults 128 / 2), so set the same values
  for training and inference of a wider/deeper model, e.g.
  `NUM_LAYERS=3 EXPORT=1 python train.py`.
- **`num_layers ≥ 2`** is required — the window reads the first hidden layer and
  the output comes off the top, so a single layer is rejected in the constructor.
- After changing the dataset or the token scheme, retrain — the jamo/symbol
  embeddings only get gradient for units that actually appear in the data.
- `../data/` contains an `안녕하세요` folder that is *not* in the `characters` list
  and is unused by the single-char loader (superseded by the export dataset).

## Performance / training speed

**Measured, not assumed.** Both datasets cache all transformed tensors in memory
(`__init__` parses every file once; `__getitem__` just returns them), so data
loading is not the bottleneck — GRU compute is.

- Single-char folders (short ~58-point sequences, batch 64): ~2.65s/epoch on M1
  MPS; data loading was only ~9% of that.
- Real export lines (long sequences, ~800 points mean, ~1400 max): **~17s/epoch**
  on MPS. The custom per-timestep `GRUCell` loop (required so the window can be
  computed mid-sequence) means there is **no fused RNN kernel** — this is the
  dominant cost and it grows with sequence length.

Levers, in rough order of impact for the line data:
- **Truncated BPTT** — the biggest lever for long lines; not yet implemented.
- **CUDA vs MPS:** CUDA helps RNNs generally, but the manual `GRUCell` loop
  forgoes cuDNN's fused kernel regardless, so expect less speedup than a plain
  `nn.GRU` would get. `get_device()` already prefers CUDA.
- `DataLoader` is single-process (`num_workers=0`); marginal given caching.

## Progress log / next experiment

Fixes landed, in order:

1. **MDN output** fixed stroke *closure* (the ㅇ no longer spirals open).
2. **Trailing-edge pen anchoring** fixed *gross composition* — strokes separate
   cleanly and the ㅏ lands to the right of the ㅇ. A representation change, no new
   inputs.
3. **Binary pen head + count-based termination** for the dropped ㅏ-tick. Replaced
   the old 3-way pen head with a binary end-of-stroke and (for single characters)
   terminate after the modal stroke count, removing the "stop early" escape hatch.
4. **Graves sliding attention window + multi-character input.**
   `GRUWithSlidingAttention` computes the window from layer 0 and feeds `w_t` back
   as input; `tokenize` produces `(U,4)` with space/punctuation as symbol tokens;
   `ExportDataset` loads real lines from the production export with a held-out-by-
   text split. Validated on synthetic stitched glyphs (the window slides from
   char 0 to char 1) before the first real-data run.
5. **Full Graves skip connections** (input→every layer, window→every layer,
   every layer→output). Diagnosed from a real-data run where XY dropped nicely but
   **pen loss was frozen at the ~1.19 prior baseline for 140 epochs**. Probing
   showed the end-of-stroke signal is present in layer 1 (~1.07) but gone by the
   top layer the pen head read (1.20) — the missing output skip. With skips the
   pen loss moves (1.21 → 1.16 in ~11 epochs). Data/segmentation were healthy
   (~8% end-of-stroke, ~65 strokes/line); it was purely architectural.
6. **Absolute-position inputs.** A converged run (2.3k epochs, val −3.21) still
   drew lines far more compressed than the human hand (w/h ~5 vs ~14), and more
   training didn't help — the drift is *information*-limited, not capacity. The
   model saw only deltas, so it never knew where it was on the line (the window's
   kappa knows only the logical character index). `HandwritingRNN` now appends the
   running absolute position — `pos0 + cumsum(deltas)`, standardized by
   `pos_mean`/`pos_std` — as `abs_pos_dim` extra input channels (2 = x,y; 1 = x;
   0 = off). Position is threaded through the state so step-by-step generation
   matches full-sequence training exactly.

**Still open:**
- Does absolute position fix the horizontal compression? (The reason for #6; judge
  on the rendered w/h ratio, not just loss.) Requires a fresh run — the input width
  changed, so old checkpoints do not load.
- Pen head is *modestly* predictive, not sharp — stroke length is highly variable
  (median 9, std 7.8), so one-step-ahead end-of-stroke is inherently partly
  stochastic (ceiling ~1.05). Feeding a timing/velocity input (recordings carry
  `timeStamp`, currently discarded) is the candidate for pushing past it.
- Long-sequence training cost — see Performance (truncated BPTT).
- Open architecture A/Bs (do after position, one at a time): LSTM vs GRU (the long
  sequences + cell-state accumulator may hold running position better), and a 3rd
  hidden layer (Graves used 3; more capacity to exploit position, low overfit risk).

**Feed timing/velocity.** The recordings carry per-dot `timeStamp` (and Ncode
gives pen dynamics), which we currently discard. Pen pauses/slow-downs physically
signal stroke ends far better than geometry, so a velocity or time-since-lift
input channel is the most promising lever for the pen head beyond the skip
connections (see the ~1.05 predictability ceiling in the progress log).

## Checkpoints

- **Architecture changed with the sliding window**, so all pre-window checkpoints
  (`best_model.mse.pt`, `best_model.mdn_v1.pt`, `best_model.mdn_v2.pt`, and older
  `best_model*.pt` / `안.pt`) **do not load** into `HandwritingRNN` anymore — the
  GRU parameter names differ and the window/symbol heads are new. They are kept
  only as historical artifacts; the window model must be retrained from scratch.
- `best_model.pt` — overwritten by the current (window + multi-char) training run.
