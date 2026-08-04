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
}

const MAX_LEN_PER_CHAR = 120; // ~58 points/char in training data; generous cap
const MAX_LEN_FLOOR = 500;
const YIELD_EVERY = 8; // macrotask break so the UI stays at 60fps

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
    const [meta, session] = await Promise.all([
      fetch(`${baseUrl}/model/model-meta.json`).then(
        (r) => r.json() as Promise<ModelMeta>,
      ),
      ort.InferenceSession.create(`${baseUrl}/model/handwriting-step.onnx`, {
        executionProviders: ["wasm"],
      }),
    ]);
    return new HandwritingModel(session, meta);
  }

  /** Stream generated pen points. Mirrors model.generate() step for step. */
  async *generate(params: GenParams, signal?: AbortSignal): AsyncGenerator<Dot> {
    const { meta } = this;
    const K = meta.numMixtures;
    const rng = new Prng(params.seed);

    const rows = tokenize(params.text);
    const U = rows.length;
    if (U === 0) return;
    const maxLen =
      params.maxLen ?? Math.max(MAX_LEN_FLOOR, U * MAX_LEN_PER_CHAR);

    const tokens = new ort.Tensor(
      "int64",
      BigInt64Array.from(rows.flat().map(BigInt)),
      [1, U, 4],
    );
    const u = new ort.Tensor(
      "float32",
      Float32Array.from({ length: U }, (_, i) => i),
      [1, 1, U],
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
      x: new ort.Tensor("float32", new Float32Array([0, 0, 0]), [1, 3]),
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

    for (let step = 0; step < maxLen; step++) {
      if (signal?.aborted) return;
      const out = await this.session.run(feeds);

      // --- end-of-stroke: greedy at temperature 0, else Bernoulli ---
      const penLogit = (out.pen_logit.data as Float32Array)[0];
      let pen: 0 | 1;
      if (params.temperature === 0) {
        pen = sigmoid(penLogit) > 0.5 ? 1 : 0;
      } else {
        pen = rng.uniform() < sigmoid(penLogit / params.temperature) ? 1 : 0;
      }

      // --- offset (dx, dy): sample the mixture, with Graves' bias ---
      // mdn_raw layout: [pi_logits(K), mu(2K), log_sigma(2K), rho_raw(K)]
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
      const mx = raw[K + 2 * k];
      const my = raw[K + 2 * k + 1];
      const sx = Math.exp(raw[3 * K + 2 * k] - params.bias);
      const sy = Math.exp(raw[3 * K + 2 * k + 1] - params.bias);
      const r = Math.tanh(raw[5 * K + k]);

      // Bivariate normal via the Cholesky factor of the covariance.
      const z1 = rng.normal();
      const z2 = rng.normal();
      const dx = mx + sx * z1;
      const dy = my + r * sy * z1 + sy * Math.sqrt(1 - r * r) * z2;

      yield { x: dx, y: dy, penState: pen };

      feeds = {
        tokens,
        u,
        mask,
        x: new ort.Tensor("float32", new Float32Array([dx, dy, pen]), [1, 3]),
        hidden: out.hidden_out as ort.Tensor,
        w: out.w_out as ort.Tensor,
        kappa: out.kappa_out as ort.Tensor,
        pos: out.pos_out as ort.Tensor,
      };

      // Terminate once the attention window slides past the last character
      // (model.generate()'s kappa test; used here for every U since the modal
      // stroke-count table isn't available in the browser).
      const kap = out.kappa_out.data as Float32Array;
      let kmean = 0;
      for (let i = 0; i < kap.length; i++) kmean += kap[i];
      if (kmean / kap.length >= U) return;

      if (step % YIELD_EVERY === YIELD_EVERY - 1) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
  }
}
