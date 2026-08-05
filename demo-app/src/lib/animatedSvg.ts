// Port of handwriting-generation/animate.py: a self-drawing, endlessly-looping
// SVG. Each stroke draws on for a slice of one shared timeline proportional to
// its point count (pen sample rate is ~fixed, so points ~ drawing time).
//
// Draw-on uses the stroke's real geometric length as the dasharray, animating
// dashoffset to 0. The hidden offset keeps one stroke-width of margin so no
// dash boundary sits on the path while hidden — otherwise the round line-cap
// would paint a phantom dot at every not-yet-drawn stroke's start.

import { Dot } from "./generator";
import {
  bounds,
  splitStrokes,
  strokeLength,
  strokeWidthFactor,
  toAbsolute,
} from "./strokes";

export interface AnimatedSvgOptions {
  secondsPerPoint?: number;
  strokeGap?: number;
  endPause?: number;
  pad?: number;
  color?: string;
}

export function toAnimatedSvg(dots: Dot[], opts: AnimatedSvgOptions = {}): string {
  const {
    secondsPerPoint = 0.02,
    strokeGap = 0.12,
    endPause = 1.2,
    pad = 0.4,
    color = "#1a1d24",
  } = opts;

  const pts = toAbsolute(dots);
  const b = bounds(pts);
  const shifted = pts.map((p) => ({ ...p, x: p.x - b.minX, y: p.y - b.minY }));
  const w = b.maxX - b.minX || 1;
  const h = b.maxY - b.minY || 1;
  const strokes = splitStrokes(shifted);
  // Pen thickness tracks LINE HEIGHT (a real pen tip is ~7% of the line),
  // not width — a width-relative pen over-inks long lines. The floor keeps
  // single flat glyphs (h ≈ 0) visible.
  const strokeWidth = Math.max(h * 0.07, Math.max(w, h) * 0.008);

  const drawTimes = strokes.map((s) => Math.max(s.length, 1) * secondsPerPoint);
  const total =
    drawTimes.reduce((a, v) => a + v, 0) +
    strokeGap * Math.max(strokes.length - 1, 0) +
    endPause;

  const els: string[] = [];
  let t = 0;
  strokes.forEach((s, i) => {
    const draw = drawTimes[i];
    const start = (t / total).toFixed(4);
    const end = ((t + draw) / total).toFixed(4);
    t += draw + strokeGap;
    const length = strokeLength(s);
    // Pressure: each stroke is drawn at its mean force's width (the dash-based
    // draw-on needs one width per path; within-stroke taper would need a
    // ribbon fill that can't dash-animate).
    const sw = strokeWidth * strokeWidthFactor(s);
    const common =
      `fill="none" stroke="${color}" stroke-width="${sw.toFixed(3)}" ` +
      'stroke-linecap="round" stroke-linejoin="round"';

    if (length < 1e-6) {
      // Single-point stroke (a dot): pop it in at its moment via opacity.
      const p = s[0];
      els.push(
        `<circle cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" ` +
          `r="${(sw * 0.6).toFixed(3)}" fill="${color}" opacity="0">` +
          `<animate attributeName="opacity" dur="${total.toFixed(3)}s" ` +
          `repeatCount="indefinite" values="0;0;1;1" ` +
          `keyTimes="0;${start};${end};1" calcMode="linear"/></circle>`,
      );
      return;
    }

    const d = s
      .map((p, j) => `${j === 0 ? "M" : "L"} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
      .join(" ");
    const gap = length + 2 * sw;
    const hidden = length + sw;
    els.push(
      `<path d="${d}" ${common} ` +
        `stroke-dasharray="${length.toFixed(3)} ${gap.toFixed(3)}" ` +
        `stroke-dashoffset="${hidden.toFixed(3)}">` +
        `<animate attributeName="stroke-dashoffset" dur="${total.toFixed(3)}s" ` +
        `repeatCount="indefinite" values="${hidden.toFixed(3)};${hidden.toFixed(3)};0;0" ` +
        `keyTimes="0;${start};${end};1" calcMode="linear"/></path>`,
    );
  });

  const vb = `${(-pad).toFixed(2)} ${(-pad).toFixed(2)} ${(w + 2 * pad).toFixed(2)} ${(h + 2 * pad).toFixed(2)}`;
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${vb}">` +
    els.join("") +
    "</svg>"
  );
}
