// Node-side driver for the single-step ONNX model: tokenizer -> ONNX loop ->
// MDN/pen sampling, producing the same Dot stream the browser produces.
// Used by smoke.ts (tests) and og-image.ts (share image).

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
// onnxruntime-node, not -web: same API, but loads its backend from disk
// instead of fetch(), which Node lacks for file URLs.
import * as ort from "onnxruntime-node";
import { Dot } from "../src/lib/generator";
import { Prng } from "../src/lib/prng";
import { tokenize } from "../src/lib/tokenizer";

const here = path.dirname(fileURLToPath(import.meta.url));
const modelDir = path.join(here, "..", "public", "model");

// Same sampling loop as HandwritingModel.generate(); duplicated here because
// the real class wires wasmPaths/fetch for the browser. Keep in sync.
export async function generateDots(
  text: string,
  seed: number,
  temperature: number,
  bias: number,
): Promise<{ dots: Dot[]; U: number; ms: number }> {
  const meta = JSON.parse(
    fs.readFileSync(path.join(modelDir, "model-meta.json"), "utf8"),
  );
  const session = await ort.InferenceSession.create(
    fs.readFileSync(path.join(modelDir, "handwriting-step.onnx")),
  );

  const K = meta.numMixtures;
  const rng = new Prng(seed);
  const rows = tokenize(text);
  const U = rows.length;

  const tokens = new ort.Tensor(
    "int64",
    BigInt64Array.from(rows.flat().map(BigInt)),
    [1, U, 4],
  );
  const u = new ort.Tensor(
    "float32",
    Float32Array.from({ length: U + 1 }, (_, i) => i),
    [1, 1, U + 1],
  );
  const mask = new ort.Tensor("float32", new Float32Array(U).fill(1), [1, U]);

  let feeds: Record<string, ort.Tensor> = {
    tokens,
    u,
    mask,
    x: new ort.Tensor("float32", new Float32Array([0, 0, 0, 0]), [1, 4]),
    hidden: new ort.Tensor(
      "float32",
      new Float32Array(meta.numLayers * meta.hiddenSize),
      [meta.numLayers, 1, meta.hiddenSize],
    ),
    w: new ort.Tensor("float32", new Float32Array(meta.windowDim), [1, meta.windowDim]),
    kappa: new ort.Tensor("float32", new Float32Array(meta.slidingWindowK), [
      1,
      meta.slidingWindowK,
    ]),
    pos: new ort.Tensor("float32", new Float32Array(2), [1, 2]),
  };

  const dots: Dot[] = [];
  const t0 = performance.now();
  let prevPen: 0 | 1 = 0;
  let parkedRun = 0;
  for (let step = 0; step < U * 120 + 500; step++) {
    const out = await session.run(feeds);

    // Termination (see generator.ts): 3 consecutive parked dots (force below
    // -2.5 standardized -- the trained end-of-text tail), or the phantom
    // past-the-end column outweighs every real character at a stroke boundary.
    if (parkedRun >= 3) break;
    if (prevPen === 1 && U > 1) {
      const phi = out.phi.data as Float32Array;
      let maxReal = -Infinity;
      for (let i = 0; i < U; i++) maxReal = Math.max(maxReal, phi[i]);
      if (phi[U] > maxReal) break;
    }

    const penLogit = (out.pen_logit.data as Float32Array)[0];
    const sig = 1 / (1 + Math.exp(-penLogit / (temperature || 1)));
    const pen: 0 | 1 =
      temperature === 0
        ? 1 / (1 + Math.exp(-penLogit)) > 0.5
          ? 1
          : 0
        : rng.uniform() < sig
          ? 1
          : 0;

    const raw = out.mdn_raw.data as Float32Array;
    let maxLogit = -Infinity;
    for (let k = 0; k < K; k++) maxLogit = Math.max(maxLogit, raw[k] * (1 + bias));
    const pi = new Float64Array(K);
    let sum = 0;
    for (let k = 0; k < K; k++) {
      pi[k] = Math.exp(raw[k] * (1 + bias) - maxLogit);
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
    const sx = Math.exp(raw[4 * K + 3 * k] - bias);
    const sy = Math.exp(raw[4 * K + 3 * k + 1] - bias);
    const sf = Math.exp(raw[4 * K + 3 * k + 2] - bias);
    const r = Math.tanh(raw[7 * K + k]);
    const z1 = rng.normal();
    const z2 = rng.normal();
    const z3 = rng.normal();
    const dx = mx + sx * z1;
    const dy = my + r * sy * z1 + sy * Math.sqrt(1 - r * r) * z2;
    const fStd = Math.min(3, Math.max(-3, mf + sf * z3));

    dots.push({
      x: dx,
      y: dy,
      penState: pen,
      f: meta.forceMean + fStd * meta.forceStd,
    });
    prevPen = pen;
    parkedRun = fStd < -2.5 ? parkedRun + 1 : 0;

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
  }
  // Drop trailing parked dots, as generator.ts does by never yielding them.
  const parkedF = meta.forceMean - 2.5 * meta.forceStd;
  while (dots.length && dots[dots.length - 1].f < parkedF) dots.pop();
  const ms = performance.now() - t0;
  return { dots, U, ms };
}
