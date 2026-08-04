// Shared stroke geometry: delta dots -> absolute points -> strokes.
// Convention (matches render.py / training): penState == 1 marks the LAST
// point of a stroke, so a new stroke starts after a pen-up point.

import { Dot } from "./generator";

export interface Pt {
  x: number;
  y: number;
  pen: number;
}

/** Deltas -> absolute points (origin at the running position's start). */
export function toAbsolute(dots: Dot[]): Pt[] {
  let x = 0;
  let y = 0;
  return dots.map((d) => {
    x += d.x;
    y += d.y;
    return { x, y, pen: d.penState };
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
