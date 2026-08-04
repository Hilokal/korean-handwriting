// Renders RecordedDots (absolute
// Ncode coordinates from the pen).
//
// Only dotType 1 (PEN_MOVE) dots carry trustworthy coordinates — DOWN/UP/HOVER
// dots can have garbage coords and act purely as stroke boundaries, matching
// the training pipeline (handwriting_dataset.py).
//
// The pen also occasionally misdecodes the Ncode pattern and emits a move dot
// with a wrong pageInfo — its coordinates live in another page's coordinate
// space and would blow up any bounding box. Cleaning therefore keeps only move
// dots on the MAJORITY page. Off-page dots are skipped as noise (without
// forcing a stroke boundary); a large off-page fraction signals a genuine
// page switch, which callers can surface via offPageCount.
//
// Two glitch filters, both matching handwriting_dataset.transform so the preview
// shows what training keeps: (1) a stray point far off the line but reached by a
// sub-MAX_DELTA step is dropped when it sits more than Y_OUTLIER from the median
// line height (a single such point renders as a long thin spike); (2) points that
// jump more than MAX_DELTA from the last *kept* point are sensor
// glitches and get dropped. The reference position carries across stroke
// boundaries (a single global running point), matching the training pipeline's
// handwriting_dataset.transform exactly — so this preview shows what training
// actually keeps. (An earlier version reset the reference at each stroke
// boundary; that let a glitch landing on a stroke's *first* point escape the
// filter and blow up the bounding box, making valid recordings render tiny —
// e.g. rec-705, rec-355. Verified against all 791 recordings: the only renders
// that change are those glitch cases plus already-broken recordings that
// training also rejects.)

import { PageInfo, RecordedDot } from "../pen/types";

const MAX_DELTA = 10.0;
const PEN_MOVE = 1;
const Y_OUTLIER = 3.0; // drop move dots more than this from the median line height

export type Point = { x: number; y: number; f: number; startNew: boolean };

export interface DotAnalysis {
  points: Point[];
  modalPage: PageInfo | null;
  moveCount: number;
  offPageCount: number;
}

const pageKey = (p: PageInfo) => `${p.section}/${p.owner}/${p.book}/${p.page}`;

function median(vals: number[]): number | null {
  if (vals.length === 0) return null;
  const s = [...vals].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function analyzeDots(dots: RecordedDot[]): DotAnalysis {
  // Majority page among move dots.
  const pageCounts = new Map<string, { page: PageInfo; n: number }>();
  let moveCount = 0;
  for (const dot of dots) {
    if (dot.dotType !== PEN_MOVE) continue;
    moveCount++;
    const key = pageKey(dot.pageInfo);
    const entry = pageCounts.get(key);
    if (entry) entry.n++;
    else pageCounts.set(key, { page: dot.pageInfo, n: 1 });
  }
  let modalPage: PageInfo | null = null;
  let modalCount = 0;
  for (const { page, n } of pageCounts.values()) {
    if (n > modalCount) {
      modalCount = n;
      modalPage = page;
    }
  }
  const modalKey = modalPage ? pageKey(modalPage) : null;

  // Median line height of on-page move dots, to drop off-line sensor spikes: a
  // stray point far from the line reached by a sub-MAX_DELTA step, which the jump
  // filter below misses. Mirrors handwriting_dataset.transform's y_outlier repair.
  const onPageYs: number[] = [];
  for (const dot of dots) {
    if (dot.dotType !== PEN_MOVE) continue;
    if (modalKey && pageKey(dot.pageInfo) !== modalKey) continue;
    onPageYs.push(dot.y);
  }
  const medY = median(onPageYs);

  const points: Point[] = [];
  let newStroke = true;
  let prevX: number | null = null;
  let prevY: number | null = null;
  for (const dot of dots) {
    if (dot.dotType !== PEN_MOVE) {
      newStroke = true; // boundary; its own coords are not ink
      continue;
    }
    if (modalKey && pageKey(dot.pageInfo) !== modalKey) {
      continue; // off-page noise: skip without breaking the stroke
    }
    if (medY !== null && Math.abs(dot.y - medY) > Y_OUTLIER) {
      continue; // off-line spike: drop the stray point, keep the stroke
    }
    if (
      prevX !== null &&
      prevY !== null &&
      (Math.abs(dot.x - prevX) > MAX_DELTA || Math.abs(dot.y - prevY) > MAX_DELTA)
    ) {
      continue; // glitch point
    }
    points.push({ x: dot.x, y: dot.y, f: dot.f ?? 0, startNew: newStroke });
    prevX = dot.x;
    prevY = dot.y;
    newStroke = false;
  }

  return { points, modalPage, moveCount, offPageCount: moveCount - modalCount };
}

export type SvgPath = {
  pathData: string;
  viewBox: string;
  width: number;
  height: number;
};

// --- Pressure-aware rendering ---------------------------------------------
//
// The pen reports tip force (`f`) on every dot, in raw sensor units whose
// scale varies by pen model/firmware. Rendering maps force to stroke width so
// a recording shows where the writer pressed. Normalization is per-recording
// and robust (5th–95th percentile) so one heavy tap can't flatten the rest,
// and force is smoothed within each stroke because the raw sensor is noisy.

const PRESSURE_SMOOTH_RADIUS = 2; // moving-average half-window, in dots
// Half-width = base * (MIN + RANGE * fnorm), so full width spans
// 0.6x–2.6x of the caller's base width across the pressure range.
const WIDTH_MIN = 0.3;
const WIDTH_RANGE = 1.0;

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const idx = (sorted.length - 1) * q;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/** Per-point pressure in 0..1: smoothed within each stroke, then normalized
 * against the recording's own 5th–95th percentile range. */
export function normalizedPressure(points: Point[]): number[] {
  const smoothed = new Array<number>(points.length);
  let strokeStart = 0;
  for (let i = 0; i <= points.length; i++) {
    if (i < points.length && !(i > 0 && points[i].startNew)) continue;
    for (let j = strokeStart; j < i; j++) {
      let sum = 0;
      let n = 0;
      const from = Math.max(strokeStart, j - PRESSURE_SMOOTH_RADIUS);
      const to = Math.min(i - 1, j + PRESSURE_SMOOTH_RADIUS);
      for (let k = from; k <= to; k++) {
        sum += points[k].f;
        n++;
      }
      smoothed[j] = sum / n;
    }
    strokeStart = i;
  }
  const sorted = [...smoothed].sort((a, b) => a - b);
  const lo = quantile(sorted, 0.05);
  const hi = quantile(sorted, 0.95);
  if (hi - lo < 1e-9) return smoothed.map(() => 0.5);
  return smoothed.map((f) => Math.min(1, Math.max(0, (f - lo) / (hi - lo))));
}

type RibbonPoint = { x: number; y: number; hw: number };

const fmt = (v: number) => v.toFixed(3);

/** Closed outline of one stroke: offset the polyline by each point's
 * half-width on both sides, with semicircular end caps. */
function strokeOutline(pts: RibbonPoint[]): string {
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

/** Build SVG path data (to be FILLED, not stroked) from raw pen dots,
 * filtered like the training pipeline, with pen pressure mapped to stroke
 * width. `baseWidthFrac` is the nominal stroke width as a fraction of the
 * drawing's height. */
export function recordedDotsToSvgPath(
  dots: RecordedDot[],
  baseWidthFrac = 1 / 100,
): SvgPath {
  const { points } = analyzeDots(dots);

  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const width = maxX - minX || 1;
  const height = maxY - minY || 1;
  const base = height * baseWidthFrac;

  const fnorm = normalizedPressure(points);
  const parts: string[] = [];
  let stroke: RibbonPoint[] = [];
  const flush = () => {
    if (stroke.length > 0) parts.push(strokeOutline(stroke));
    stroke = [];
  };
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (p.startNew) flush();
    const x = p.x - minX;
    const y = p.y - minY;
    const hw = base * (WIDTH_MIN + WIDTH_RANGE * fnorm[i]);
    const last = stroke[stroke.length - 1];
    if (last && Math.abs(last.x - x) < 1e-6 && Math.abs(last.y - y) < 1e-6) {
      last.hw = Math.max(last.hw, hw); // duplicate coordinate; keep widest
      continue;
    }
    stroke.push({ x, y, hw });
  }
  flush();

  return { pathData: parts.join(" "), viewBox: `0 0 ${width} ${height}`, width, height };
}
