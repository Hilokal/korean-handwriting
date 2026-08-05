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

/** Pressure -> stroke-width multiplier around the nominal width.
 *
 * Calibrated to the collection app's strokeSvg mapping (5th-95th force
 * percentile -> 0.3-1.3x base, so the MEDIAN stroke is 0.8x): the corpus
 * force distribution spans ~±2.32 sigma, so its percentile window becomes a
 * fixed linear ramp in standardized force — streaming-friendly, no
 * per-drawing normalization pass needed. */
export function pressureWidthFactor(fRaw: number): number {
  const z = (fRaw - FORCE_MEAN) / FORCE_STD;
  return Math.min(1.3, Math.max(0.3, 0.8 + 0.215 * z));
}

/** Mean width factor over one stroke's points. (Used by the animated SVG,
 * whose dash-based draw-on needs a single width per path; the live canvas
 * uses strokeRibbonPath for true per-point width.) */
export function strokeWidthFactor(stroke: Pt[]): number {
  if (stroke.length === 0) return 1;
  let s = 0;
  for (const p of stroke) s += pressureWidthFactor(p.f);
  return s / stroke.length;
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

const SMOOTH_RADIUS = 2; // moving-average half-window, in points (per stroke)

/** One stroke -> filled-ribbon path data with per-point pressure width.
 * `baseWidth` is the nominal stroke width (a diameter, like stroke-width);
 * force is smoothed within the stroke before mapping — the raw sensor (and
 * the model's samples) are noisy point to point. */
export function strokeRibbonPath(stroke: Pt[], baseWidth: number): string {
  const ribbon: RibbonPoint[] = [];
  for (let j = 0; j < stroke.length; j++) {
    const from = Math.max(0, j - SMOOTH_RADIUS);
    const to = Math.min(stroke.length - 1, j + SMOOTH_RADIUS);
    let f = 0;
    for (let k = from; k <= to; k++) f += stroke[k].f;
    const hw = (baseWidth * pressureWidthFactor(f / (to - from + 1))) / 2;
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
