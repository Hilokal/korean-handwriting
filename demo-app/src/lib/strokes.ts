// Shared stroke geometry: delta dots -> absolute points -> strokes.
// Convention (matches render.py / training): penState == 1 marks the LAST
// point of a stroke, so a new stroke starts after a pen-up point.

import { Dot } from "./generator";

export interface Pt {
  x: number;
  y: number;
  pen: number;
  f: number; // pen force, raw sensor units
}

// Pen-force standardization constants — the frozen convention shared with the
// training pipeline (handwriting_dataset.py FORCE_MEAN/FORCE_STD). Used only
// to map force to a width factor; do not recompute per export.
export const FORCE_MEAN = 521.4;
export const FORCE_STD = 138.7;

// Width window: the 5th-95th percentile of the DRAWING'S OWN force range maps
// to 0.3x-1.3x of the base width (collection-app strokeSvg / render.py
// convention). Per-drawing normalization matters for generated ink: sampling
// bias compresses the model's force range, and a fixed corpus-wide ramp left
// biased drawings looking near-uniform — percentiles stretch whatever
// dynamic range a drawing has to the full window.
export const WIDTH_MIN = 0.3;
export const WIDTH_RANGE = 1.0;

const SMOOTH_RADIUS = 2; // moving-average half-window, in points (per stroke)

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const idx = (sorted.length - 1) * q;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/** Per-point pressure in 0..1 for a whole drawing: smoothed within each
 * stroke (the sensor and the model's samples are noisy point to point), then
 * normalized against the drawing's own 5th-95th percentile range. Ported
 * from collection-app strokeSvg.ts (normalizedPressure) — keep in sync. */
export function normalizedPressure(strokes: Pt[][]): number[][] {
  const smoothed = strokes.map((stroke) => {
    const row = new Array<number>(stroke.length);
    for (let j = 0; j < stroke.length; j++) {
      const from = Math.max(0, j - SMOOTH_RADIUS);
      const to = Math.min(stroke.length - 1, j + SMOOTH_RADIUS);
      let s = 0;
      for (let k = from; k <= to; k++) s += stroke[k].f;
      row[j] = s / (to - from + 1);
    }
    return row;
  });
  const all = smoothed.flat().sort((a, b) => a - b);
  const lo = quantile(all, 0.05);
  const hi = quantile(all, 0.95);
  if (hi - lo < 1e-9) return smoothed.map((r) => r.map(() => 0.5));
  return smoothed.map((r) =>
    r.map((f) => Math.min(1, Math.max(0, (f - lo) / (hi - lo)))),
  );
}

/** Mean width factor for one stroke given its normalized-pressure row. (Used
 * by the animated SVG, whose dash-based draw-on needs a single width per
 * path; the live canvas uses strokeRibbonPath for true per-point width.) */
export function strokeWidthFactor(fnorm: number[]): number {
  if (fnorm.length === 0) return 1;
  let s = 0;
  for (const v of fnorm) s += v;
  return WIDTH_MIN + WIDTH_RANGE * (s / fnorm.length);
}

export type RibbonPoint = { x: number; y: number; hw: number };

const fmt = (v: number) => v.toFixed(3);

/** Closed outline of one stroke: offset the polyline by each point's
 * half-width on both sides, with semicircular end caps. Ported from
 * collection-app strokeSvg.ts (strokeOutline) — keep in sync. */
export function strokeOutline(pts: RibbonPoint[]): string {
  if (pts.length === 1) {
    const { x, y, hw } = pts[0];
    const r = fmt(hw);
    return (
      `M ${fmt(x - hw)} ${fmt(y)} ` +
      `A ${r} ${r} 0 1 0 ${fmt(x + hw)} ${fmt(y)} ` +
      `A ${r} ${r} 0 1 0 ${fmt(x - hw)} ${fmt(y)} Z`
    );
  }
  const left: string[] = [];
  const right: string[] = [];
  let tx = 1;
  let ty = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[Math.max(0, i - 1)];
    const b = pts[Math.min(pts.length - 1, i + 1)];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    if (len > 1e-9) {
      tx = dx / len;
      ty = dy / len;
    }
    // Normal to the (smoothed) tangent; reuse the previous tangent on
    // zero-length steps so the normal never degenerates.
    const p = pts[i];
    left.push(`${fmt(p.x - ty * p.hw)} ${fmt(p.y + tx * p.hw)}`);
    right.push(`${fmt(p.x + ty * p.hw)} ${fmt(p.y - tx * p.hw)}`);
  }
  const rEnd = fmt(pts[pts.length - 1].hw);
  const rStart = fmt(pts[0].hw);
  return (
    `M ${left[0]} L ${left.slice(1).join(" L ")} ` +
    `A ${rEnd} ${rEnd} 0 0 0 ${right[right.length - 1]} ` +
    `L ${right.slice(0, -1).reverse().join(" L ")} ` +
    `A ${rStart} ${rStart} 0 0 0 ${left[0]} Z`
  );
}

/** One stroke -> filled-ribbon path data with per-point pressure width.
 * `fnorm` is the stroke's row from normalizedPressure(); `baseWidth` is the
 * nominal stroke width (a diameter, like stroke-width). */
export function strokeRibbonPath(
  stroke: Pt[],
  fnorm: number[],
  baseWidth: number,
): string {
  const ribbon: RibbonPoint[] = [];
  for (let j = 0; j < stroke.length; j++) {
    const hw = (baseWidth * (WIDTH_MIN + WIDTH_RANGE * fnorm[j])) / 2;
    const p = stroke[j];
    const last = ribbon[ribbon.length - 1];
    if (last && Math.abs(last.x - p.x) < 1e-6 && Math.abs(last.y - p.y) < 1e-6) {
      last.hw = Math.max(last.hw, hw); // duplicate coordinate; keep widest
      continue;
    }
    ribbon.push({ x: p.x, y: p.y, hw });
  }
  return strokeOutline(ribbon);
}

/** Deltas -> absolute points (origin at the running position's start). */
export function toAbsolute(dots: Dot[]): Pt[] {
  let x = 0;
  let y = 0;
  return dots.map((d) => {
    x += d.x;
    y += d.y;
    return { x, y, pen: d.penState, f: d.f ?? FORCE_MEAN };
  });
}

/** Group absolute points into strokes (pen == 1 ends a stroke). */
export function splitStrokes(pts: Pt[]): Pt[][] {
  const strokes: Pt[][] = [];
  let cur: Pt[] = [];
  for (const p of pts) {
    cur.push(p);
    if (p.pen >= 0.5) {
      strokes.push(cur);
      cur = [];
    }
  }
  if (cur.length) strokes.push(cur);
  return strokes;
}

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export function bounds(pts: Pt[]): Bounds {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return { minX, minY, maxX, maxY };
}

export function strokeLength(stroke: Pt[]): number {
  let len = 0;
  for (let i = 1; i < stroke.length; i++) {
    len += Math.hypot(stroke[i].x - stroke[i - 1].x, stroke[i].y - stroke[i - 1].y);
  }
  return len;
}
