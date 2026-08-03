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

export type Point = { x: number; y: number; startNew: boolean };

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
    points.push({ x: dot.x, y: dot.y, startNew: newStroke });
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

/** Build SVG path data from raw pen dots, filtered like the training pipeline. */
export function recordedDotsToSvgPath(dots: RecordedDot[]): SvgPath {
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

  const pathData = points
    .map((p, i) => {
      const cmd = i === 0 || p.startNew ? "M" : "L";
      return `${cmd} ${(p.x - minX).toFixed(3)} ${(p.y - minY).toFixed(3)}`;
    })
    .join(" ");

  return { pathData, viewBox: `0 0 ${width} ${height}`, width, height };
}
