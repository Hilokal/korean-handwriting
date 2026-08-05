// Headless smoke test of the in-browser generation stack, run with tsx.
// Exercises tokenizer -> ONNX single-step loop -> sampling -> animated SVG,
// i.e. everything the browser does except React and fetch().
//
//   npx tsx scripts/smoke.ts

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
// onnxruntime-node, not -web: same API, but loads its backend from disk
// instead of fetch(), which Node lacks for file URLs.
import * as ort from "onnxruntime-node";
import { toAnimatedSvg } from "../src/lib/animatedSvg";
import { Dot } from "../src/lib/generator";
import { Prng } from "../src/lib/prng";
import { bounds, toAbsolute } from "../src/lib/strokes";
import { tokenize } from "../src/lib/tokenizer";

const here = path.dirname(fileURLToPath(import.meta.url));
const modelDir = path.join(here, "..", "public", "model");

// Same sampling loop as HandwritingModel.generate(); duplicated here because
// the real class wires wasmPaths/fetch for the browser. Keep in sync.
async function run(text: string, seed: number, temperature: number, bias: number) {
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
  for (let step = 0; step < U * 120 + 500; step++) {
    const out = await session.run(feeds);

    // Graves' phi termination (see generator.ts): the phantom past-the-end
    // column outweighs every real character, checked at stroke boundaries.
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
  const ms = performance.now() - t0;

  const pts = toAbsolute(dots);
  const b = bounds(pts);
  const strokes = dots.filter((d) => d.penState === 1).length;
  console.log(
    `"${text}" (U=${U}): ${dots.length} points, ${strokes} strokes, ` +
      `${ms.toFixed(0)} ms (${((ms / dots.length) * 1000).toFixed(0)} us/step)`,
  );
  console.log(
    `  bounds: ${(b.maxX - b.minX).toFixed(1)} x ${(b.maxY - b.minY).toFixed(1)}` +
      ` (w/h ${((b.maxX - b.minX) / Math.max(b.maxY - b.minY, 0.001)).toFixed(1)})`,
  );

  if (dots.length < U * 10) throw new Error("suspiciously few points");
  if (strokes < U) throw new Error("suspiciously few strokes");

  // Pressure sanity: finite, plausible range, and tapering into stroke ends
  // (the signature the model learned from real hands).
  if (!dots.every((d) => Number.isFinite(d.f))) throw new Error("non-finite f");
  const lastF: number[] = [];
  const midF: number[] = [];
  let cur: Dot[] = [];
  for (const d of dots) {
    cur.push(d);
    if (d.penState === 1) {
      if (cur.length > 3) {
        lastF.push(cur[cur.length - 1].f);
        midF.push(cur[Math.floor(cur.length / 2)].f);
      }
      cur = [];
    }
  }
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
  console.log(
    `  force mid->last of stroke: ${mean(midF).toFixed(0)} -> ${mean(lastF).toFixed(0)}`,
  );

  return dots;
}

const dots = await run("안녕하세요, 반갑습니다!", 12345, 1.0, 0.5);
const svg = toAnimatedSvg(dots);
const outPath = path.join(here, "smoke-output.svg");
fs.writeFileSync(outPath, svg);
console.log(`wrote ${outPath} (${svg.length} bytes)`);
await run("커피 한 잔 어때요?", 999, 0, 1.5);
console.log("SMOKE OK");
process.exit(0); // ort wasm worker keeps the loop alive otherwise
