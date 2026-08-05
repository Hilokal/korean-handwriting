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

/** Mean width factor over one stroke's points. */
export function strokeWidthFactor(stroke: Pt[]): number {
  if (stroke.length === 0) return 1;
  let s = 0;
  for (const p of stroke) s += pressureWidthFactor(p.f);
  return s / stroke.length;
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
