#!/usr/bin/env python3
"""Render a delta-encoded stroke sequence (generated.json or a recording) to SVG.

generated.json has the shape {"dots": [{"x": dx, "y": dy, "penState": p, "f": force}, ...]}
where (x, y) are *deltas* and penState == 1 marks the last point of a stroke
(pen lifts after it, Graves-style trailing edge). This mirrors the filtering and
stroke-break convention used by the training pipeline (handwriting_dataset.py).

If dots carry `f` (pen force, raw units), it is mapped to stroke width the same
way the collection app's strokeSvg.ts does: smoothed within each stroke (the
sensor is noisy), normalized per-drawing against the 5th-95th percentile (so one
heavy tap can't flatten the rest), then width = base * (0.3 + 1.0 * fnorm).
Dots without `f` render at the constant base width, as before.

    python render.py generated.json out.svg
"""
import json
import sys

# Match collection-app/client/src/components/strokeSvg.ts
PRESSURE_SMOOTH_RADIUS = 2  # moving-average half-window, in dots
WIDTH_MIN = 0.3
WIDTH_RANGE = 1.0


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _normalized_pressure(strokes):
    """Per-point pressure in 0..1 for a list of strokes [[(x, y, f), ...], ...]:
    smoothed within each stroke, normalized against the drawing's own
    5th-95th percentile range."""
    smoothed = []
    for stroke in strokes:
        fs = [p[2] for p in stroke]
        row = []
        for j in range(len(fs)):
            lo = max(0, j - PRESSURE_SMOOTH_RADIUS)
            hi = min(len(fs) - 1, j + PRESSURE_SMOOTH_RADIUS)
            row.append(sum(fs[lo : hi + 1]) / (hi - lo + 1))
        smoothed.append(row)

    all_f = sorted(f for row in smoothed for f in row)
    lo = _quantile(all_f, 0.05)
    hi = _quantile(all_f, 0.95)
    if hi - lo < 1e-9:
        return [[0.5] * len(row) for row in smoothed]
    return [[min(1.0, max(0.0, (f - lo) / (hi - lo))) for f in row] for row in smoothed]


def strokes_to_svg(dots, stroke_width=1.6):
    # deltas -> absolute positions, normalized so the min corner is the origin
    x = y = 0.0
    pts = []
    for d in dots:
        x += float(d["x"])
        y += float(d["y"])
        pts.append((x, y, float(d.get("penState", 0)), d.get("f")))
    minx = min(p[0] for p in pts)
    miny = min(p[1] for p in pts)
    pts = [(px - minx, py - miny, pen, f) for px, py, pen, f in pts]
    w = max(p[0] for p in pts) or 1.0
    h = max(p[1] for p in pts) or 1.0

    # A point starts a new stroke when the *previous* point was end-of-stroke.
    strokes = []
    cur = []
    for i, (px, py, pen, f) in enumerate(pts):
        if (i == 0 or pts[i - 1][2] >= 0.5) and cur:
            strokes.append(cur)
            cur = []
        cur.append((px, py, f))
    if cur:
        strokes.append(cur)

    style = (
        'fill="none" stroke="#1a1d24" vector-effect="non-scaling-stroke" '
        'stroke-linecap="round" stroke-linejoin="round"'
    )

    have_f = all(p[3] is not None for p in pts)
    if not have_f:
        # Constant width: one path, M at each stroke start.
        cmds = []
        for stroke in strokes:
            for j, (px, py, _f) in enumerate(stroke):
                cmds.append(f"{'M' if j == 0 else 'L'} {px:.2f} {py:.2f}")
        body = f'<path d="{" ".join(cmds)}" stroke-width="{stroke_width}" {style}/>'
    else:
        # Pressure-aware: one path per segment, width from the mean of its
        # endpoints' normalized pressure. (vector-effect widths are in screen
        # px, so widths stay meaningful regardless of the viewBox scale.)
        fnorm = _normalized_pressure(strokes)
        parts = []
        for stroke, frow in zip(strokes, fnorm):
            for j in range(1, len(stroke)):
                (x1, y1, _), (x2, y2, _) = stroke[j - 1], stroke[j]
                fn = (frow[j - 1] + frow[j]) / 2
                sw = stroke_width * (WIDTH_MIN + WIDTH_RANGE * fn)
                parts.append(
                    f'<path d="M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}" '
                    f'stroke-width="{sw:.2f}" {style}/>'
                )
            if len(stroke) == 1:  # dot-only stroke: render as a zero-length cap
                (x1, y1, _) = stroke[0]
                sw = stroke_width * (WIDTH_MIN + WIDTH_RANGE * frow[0])
                parts.append(
                    f'<path d="M {x1:.2f} {y1:.2f} L {x1:.2f} {y1:.2f}" '
                    f'stroke-width="{sw:.2f}" {style}/>'
                )
        body = "".join(parts)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {h:.2f}">'
        f"{body}</svg>"
    )


def main():
    if len(sys.argv) != 3:
        print("usage: python render.py <input.json> <output.svg>", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = json.load(f)
    svg = strokes_to_svg(data["dots"])
    with open(sys.argv[2], "w") as f:
        f.write(svg)
    print(f"wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
