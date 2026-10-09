// Headless smoke test of the in-browser generation stack, run with tsx.
// Exercises tokenizer -> ONNX single-step loop -> sampling -> animated SVG,
// i.e. everything the browser does except React and fetch().
//
//   npx tsx scripts/smoke.ts

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { toAnimatedSvg } from "../src/lib/animatedSvg";
import { Dot } from "../src/lib/generator";
import { bounds, toAbsolute } from "../src/lib/strokes";
import { generateDots } from "./generate-node";

const here = path.dirname(fileURLToPath(import.meta.url));

// Sampling loop lives in generate-node.ts (shared with og-image.ts).
async function run(text: string, seed: number, temperature: number, bias: number) {
  const { dots, U, ms } = await generateDots(text, seed, temperature, bias);

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
