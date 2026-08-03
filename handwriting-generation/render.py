#!/usr/bin/env python3
"""Render a delta-encoded stroke sequence (generated.json or a recording) to SVG.

generated.json has the shape {"dots": [{"x": dx, "y": dy, "penState": p}, ...]}
where (x, y) are *deltas* and penState == 1 marks the last point of a stroke
(pen lifts after it, Graves-style trailing edge). This mirrors the filtering and
stroke-break convention used by the training pipeline (handwriting_dataset.py).

    python render.py generated.json out.svg
"""
import json
import sys


def strokes_to_svg(dots, stroke_width=1.6):
    # deltas -> absolute positions, normalized so the min corner is the origin
    x = y = 0.0
    pts = []
    for d in dots:
        x += float(d["x"])
        y += float(d["y"])
        pts.append((x, y, float(d.get("penState", 0))))
    minx = min(p[0] for p in pts)
    miny = min(p[1] for p in pts)
    pts = [(px - minx, py - miny, pen) for px, py, pen in pts]
    w = max(p[0] for p in pts) or 1.0
    h = max(p[1] for p in pts) or 1.0

    # A point starts a new stroke (M) when the *previous* point was end-of-stroke.
    cmds = []
    for i, (px, py, _pen) in enumerate(pts):
        start_new = i == 0 or pts[i - 1][2] >= 0.5
        cmds.append(f"{'M' if start_new else 'L'} {px:.2f} {py:.2f}")
    path = " ".join(cmds)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {h:.2f}">'
        f'<path d="{path}" fill="none" stroke="#1a1d24" stroke-width="{stroke_width}" '
        f'vector-effect="non-scaling-stroke" stroke-linecap="round" '
        f'stroke-linejoin="round"/></svg>'
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
