# demo-app — hangul.ink

Public demo site for the Korean handwriting model. The **model runs entirely in
the browser**: the single-step ONNX export of the GRU (1.3 MB) is driven by a
TypeScript autoregressive loop (MDN sampling, pen-lift sampling, attention-window
termination), streaming points into a live pen animation. No inference server.

- English/Korean UI (auto-detected, switchable)
- Temperature / bias / seed controls — a (text, params, seed, model version)
  tuple reproduces the exact strokes
- Downloads: self-drawing animated SVG and the raw stroke JSON
- 👍/👎 feedback posts to the collection app (`/api/feedback`), where the admin
  UI can promote a failed sentence into the writers' assignment pool

## Development

```bash
npm install
npm run dev        # http://localhost:5174
```

The model files are committed at `public/model/` (`handwriting-step.onnx` +
`model-meta.json`). After retraining, regenerate them:

```bash
cd ../handwriting-generation
./venv/bin/python export_onnx.py --model best_model.emb8.pt \
    --num-layers 3 --embedding-size 8 --out-dir ../demo-app/public/model
./venv/bin/python test_onnx_parity.py --model best_model.emb8.pt \
    --num-layers 3 --embedding-size 8   # must print PARITY OK
```

The `version` field in `model-meta.json` (hash of the ONNX file) is stamped
into every feedback record, so feedback stays attributable across retrains.

Headless smoke test of the whole generation stack (no browser needed):

```bash
npx tsx scripts/smoke.ts
```

## Deploying (Cloudflare Workers static assets)

```bash
npx wrangler login   # once
npm run deploy       # builds, then wrangler deploy
```

One-time dashboard setup:

1. Add `hangul.ink` (and `hangeul.ink`) as Cloudflare zones; point the
   registrar's nameservers at Cloudflare.
2. Workers & Pages → `hangul-ink` → Settings → Domains & Routes → add custom
   domain `hangul.ink` (and `www.hangul.ink` if desired).
3. For `hangeul.ink`: Bulk Redirects (or a Redirect Rule on that zone) →
   301 everything to `https://hangul.ink`.

## Feedback endpoint

Feedback posts to the collection app on Fly
(`https://handwriting-collection.fly.dev/api/feedback` by default; override
with `VITE_FEEDBACK_API` at build time). CORS on that endpoint allowlists
`hangul.ink` / `hangeul.ink` / `localhost:5174` — extend via the
`FEEDBACK_ORIGINS` env var on the Fly app. Deploy the collection app once to
pick up the endpoint + `demo_feedback` migration (`fly deploy`).

## Notes

- `public/ort/` is generated at build time (wasm runtime copied from
  `node_modules/onnxruntime-web`) and gitignored.
- Single characters terminate via the attention window like full sentences —
  the modal-stroke-count table used by `inference.py` needs training data,
  which the browser doesn't have. Occasionally a lone glyph may run long;
  sentences are unaffected.
- First-visit payload ≈ 4.8 MB over the wire (wasm 3.5 MB gzipped + model
  1.3 MB); everything is cached after that.
