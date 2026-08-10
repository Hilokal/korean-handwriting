// In-browser port of HandwritingRNN.generate() (handwriting-generation/model.py).
// The single-step recurrence runs as an ONNX graph; this file owns the
// autoregressive loop: MDN sampling, pen-lift sampling, and termination.
// Keep the sampling math in lockstep with model.generate().

// The /wasm entry is the CPU-only build: it skips the 27 MB WebGPU (jsep)
// backend that the default entry would pull into the bundle.
import * as ort from "onnxruntime-web/wasm";
import { Prng } from "./prng";
import { tokenize } from "./tokenizer";

export interface ModelMeta {
  version: string;
  checkpoint: string;
  hiddenSize: number;
  numLayers: number;
  embeddingSize: number;
  numMixtures: number;
  windowDim: number;
  slidingWindowK: number;
  absPosDim: number;
  inputSize: number; // 4: [dx, dy, penState, f]
  forceMean: number; // pen-force standardization (frozen; see training CLAUDE.md)
  forceStd: number;
  phantomPhi: boolean; // u grid is U+1 long; phi's last column = past-the-end
  eotSymbolId: number; // tokenizer appends this trained end-of-text unit
}

export interface GenParams {
  text: string;
  temperature: number; // end-of-stroke sigmoid temperature; 0 = greedy
  bias: number; // MDN bias; higher = cleaner/less varied strokes
  seed: number;
  maxLen?: number;
}

export interface Dot {
  x: number; // dx (delta-encoded, like generated.json)
  y: number; // dy
  penState: 0 | 1; // 1 = last point of a stroke
  f: number; // pen force, raw sensor units (same scale as the recordings)
}

const MAX_LEN_PER_CHAR = 120; // ~58 points/char in training data; generous cap
const MAX_LEN_FLOOR = 500;
const YIELD_EVERY = 8; // macrotask break so the UI stays at 60fps
// The trained parked-pen tail sits at standardized force -3.0; every real
// point is above ~-2.3. Dots below this are the model parking after the text:
// buffer them instead of yielding, so they vanish if generation ends there
// (the streaming equivalent of model.generate()'s trailing-force trim).
const PARKED_F_STD = -2.5;

function sigmoid(v: number): number {
  return 1 / (1 + Math.exp(-v));
}

export class HandwritingModel {
  private constructor(
    private session: ort.InferenceSession,
    public readonly meta: ModelMeta,
  ) {}

  static async load(baseUrl = ""): Promise<HandwritingModel> {
    // Fully-qualified URL, not a root-relative path: ort loads its runtime
    // via dynamic import(), and Vite's dev server refuses module imports that
    // resolve into /public — an external (absolute) URL bypasses that.
    ort.env.wasm.wasmPaths = new URL(`${baseUrl}/ort/`, location.href).href;
    ort.env.wasm.numThreads = 1; // single-thread wasm avoids COOP/COEP headers
    // model-meta.json and handwriting-step.onnx keep stable filenames across
    // deploys, so the CDN edge can serve a stale pair after a model update
    // (observed 2026-08-11: fresh app shell, stale meta). Bust the tiny meta
    // fetch on every load, then key the big onnx URL by the model version it
    // announces -- each model gets its own cache entry, and the two can never
    // disagree.
    const meta = (await fetch(
      `${baseUrl}/model/model-meta.json?t=${Date.now()}`,
    ).then((r) => r.json())) as ModelMeta;
    const session = await ort.InferenceSession.create(
      `${baseUrl}/model/handwriting-step.onnx?v=${meta.version}`,
      { executionProviders: ["wasm"] },
    );
    return new HandwritingModel(session, meta);
  }

  /** Stream generated pen points. Mirrors model.generate() step for step. */
  async *generate(params: GenParams, signal?: AbortSignal): AsyncGenerator<Dot> {
    const { meta } = this;
    const K = meta.numMixtures;
    const rng = new Prng(params.seed);

    const rows = tokenize(params.text);
    const U = rows.length; // characters + the EOT unit tokenize() appends
    if (U <= 1) return; // only the EOT row: empty text
    const maxLen =
      params.maxLen ?? Math.max(MAX_LEN_FLOOR, U * MAX_LEN_PER_CHAR);

    const tokens = new ort.Tensor(
      "int64",
      BigInt64Array.from(rows.flat().map(BigInt)),
      [1, U, 4],
    );
    // U+1 indices: the last is the phantom past-the-end position whose phi
    // weight drives Graves' termination test (matches model.generate()).
    const u = new ort.Tensor(
      "float32",
      Float32Array.from({ length: U + 1 }, (_, i) => i),
      [1, 1, U + 1],
    );
    const mask = new ort.Tensor(
      "float32",
      new Float32Array(U).fill(1),
      [1, U],
    );

    let feeds: Record<string, ort.Tensor> = {
      tokens,
      u,
      mask,
      // [dx, dy, penState, f]; f = 0 is the corpus-mean force, standardized.
      x: new ort.Tensor("float32", new Float32Array([0, 0, 0, 0]), [1, 4]),
      hidden: new ort.Tensor(
        "float32",
        new Float32Array(meta.numLayers * meta.hiddenSize),
        [meta.numLayers, 1, meta.hiddenSize],
      ),
      w: new ort.Tensor("float32", new Float32Array(meta.windowDim), [
        1,
        meta.windowDim,
      ]),
      kappa: new ort.Tensor("float32", new Float32Array(meta.slidingWindowK), [
        1,
        meta.slidingWindowK,
      ]),
      pos: new ort.Tensor("float32", new Float32Array(2), [1, 2]),
    };

    let prevPen: 0 | 1 = 0;
    const parked: Dot[] = []; // see PARKED_F_STD
    for (let step = 0; step < maxLen; step++) {
      if (signal?.aborted) return;
      const out = await this.session.run(feeds);

      // --- termination (attention-based, mirrors model.generate()) ---
      // This run consumed the previously yielded dot; if that dot ended a
      // stroke, the text is finished when either (a) the window's peak sits
      // on the final unit -- the trained EOT token, which owns no strokes, so
      // this can't truncate a multi-stroke final syllable -- or (b) the
      // phantom past-the-end phi outweighs every real unit (backstop).
      // Checking before sampling keeps the timing identical to the Python loop.
      if (prevPen === 1 && U > 1) {
        const phi = out.phi.data as Float32Array; // (1, U+1)
        let maxReal = -Infinity;
        let argmax = 0;
        for (let i = 0; i < U; i++)
          if (phi[i] > maxReal) {
            maxReal = phi[i];
            argmax = i;
          }
        if (argmax === U - 1 || phi[U] > maxReal) return;
      }

      // --- end-of-stroke: greedy at temperature 0, else Bernoulli ---
      const penLogit = (out.pen_logit.data as Float32Array)[0];
      let pen: 0 | 1;
      if (params.temperature === 0) {
        pen = sigmoid(penLogit) > 0.5 ? 1 : 0;
      } else {
        pen = rng.uniform() < sigmoid(penLogit / params.temperature) ? 1 : 0;
      }

      // --- offset (dx, dy) + pressure f: sample the mixture, with Graves' bias ---
      // mdn_raw layout: [pi_logits(K), mu(3K), log_sigma(3K), rho_raw(K)];
      // per component mu/sigma are (dx, dy, f) triples. f is the next point's
      // ABSOLUTE standardized force (not a delta); it shares the component
      // pick k with the offset, so pressure is mode-dependent.
      const raw = out.mdn_raw.data as Float32Array;
      let maxLogit = -Infinity;
      for (let k = 0; k < K; k++) {
        const v = raw[k] * (1 + params.bias);
        if (v > maxLogit) maxLogit = v;
      }
      let sum = 0;
      const pi = new Float64Array(K);
      for (let k = 0; k < K; k++) {
        pi[k] = Math.exp(raw[k] * (1 + params.bias) - maxLogit);
        sum += pi[k];
      }
      let pick = rng.uniform() * sum;
      let k = 0;
      for (; k < K - 1; k++) {
        pick -= pi[k];
        if (pick <= 0) break;
      }
      const mx = raw[K + 3 * k];
      const my = raw[K + 3 * k + 1];
      const mf = raw[K + 3 * k + 2];
      const sx = Math.exp(raw[4 * K + 3 * k] - params.bias);
      const sy = Math.exp(raw[4 * K + 3 * k + 1] - params.bias);
      const sf = Math.exp(raw[4 * K + 3 * k + 2] - params.bias);
      const r = Math.tanh(raw[7 * K + k]);

      // Bivariate normal via the Cholesky factor of the covariance; pressure
      // is independent within the component. Clamp f to ~the data range so
      // the fed-back input stays in-distribution (matches model.generate()).
      const z1 = rng.normal();
      const z2 = rng.normal();
      const z3 = rng.normal();
      const dx = mx + sx * z1;
      const dy = my + r * sy * z1 + sy * Math.sqrt(1 - r * r) * z2;
      const fStd = Math.min(3, Math.max(-3, mf + sf * z3));

      const dot: Dot = {
        x: dx,
        y: dy,
        penState: pen,
        f: meta.forceMean + fStd * meta.forceStd,
      };
      // Parked dots are still fed back to the model (feeds below use dx/dy/
      // pen/fStd regardless) -- buffering only affects what gets displayed.
      if (fStd < PARKED_F_STD) {
        parked.push(dot);
      } else {
        yield* parked.splice(0);
        yield dot;
      }
      prevPen = pen;

      feeds = {
        tokens,
        u,
        mask,
        x: new ort.Tensor(
          "float32",
          new Float32Array([dx, dy, pen, fStd]),
          [1, 4],
        ),
        hidden: out.hidden_out as ort.Tensor,
        w: out.w_out as ort.Tensor,
        kappa: out.kappa_out as ort.Tensor,
        pos: out.pos_out as ort.Tensor,
      };

      if (step % YIELD_EVERY === YIELD_EVERY - 1) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
  }
}
