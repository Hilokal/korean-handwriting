#!/usr/bin/env python3
"""Render a delta-encoded stroke sequence to an *animated* SVG that draws itself.

Same input as render.py -- {"dots": [{"x": dx, "y": dy, "penState": p}, ...]}
with (x, y) deltas and penState == 1 marking the last point of a stroke. Instead
of a static path, each stroke is drawn on progressively, in order, so the result
plays back like watching the pen write. The animation loops forever.

    python animate.py generated.json out.svg
    python animate.py generated.json out.svg --seconds-per-point 0.04

The pen is sampled at a roughly fixed rate, so a stroke's point count is a good
proxy for how long it took to draw -- we spend `--seconds-per-point` of animation
time per point, which keeps the playback speed true to the original hand.

The "draw-on" effect uses each stroke's real geometric length as the
stroke-dasharray, animating stroke-dashoffset from that length down to 0. (The
tempting pathLength="1" normalization trick is avoided on purpose: it is ignored
under vector-effect="non-scaling-stroke", which leaves every stroke fully drawn.)
"""
import argparse
import json
from math import hypot


def _abs_points(dots):
    """Deltas -> absolute points, normalized so the min corner is the origin."""
    x = y = 0.0
    pts = []
    for d in dots:
        x += float(d["x"])
        y += float(d["y"])
        pts.append((x, y, float(d.get("penState", 0))))
    minx = min(p[0] for p in pts)
    miny = min(p[1] for p in pts)
    return [(px - minx, py - miny, pen) for px, py, pen in pts]


def _split_strokes(pts):
    """Group points into strokes; a stroke ends at a point with penState >= 0.5."""
    strokes, cur = [], []
    for p in pts:
        cur.append(p)
        if p[2] >= 0.5:
            strokes.append(cur)
            cur = []
    if cur:
        strokes.append(cur)
    return strokes


def _length(stroke):
    return sum(
        hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(stroke, stroke[1:])
    )


def strokes_to_animated_svg(
    dots,
    seconds_per_point=0.02,
    stroke_gap=0.12,
    end_pause=1.2,
    pad=0.4,
):
    pts = _abs_points(dots)
    w = max(p[0] for p in pts) or 1.0
    h = max(p[1] for p in pts) or 1.0
    strokes = _split_strokes(pts)
    stroke_width = max(w, h) * 0.02  # pen thickness relative to the drawing size

    # One timeline for the whole cycle: each stroke draws for a slice proportional
    # to its point count, with a short pen-lift gap between strokes and a hold at
    # the end before the loop restarts.
    draw_times = [max(len(s), 1) * seconds_per_point for s in strokes]
    total = sum(draw_times) + stroke_gap * max(len(strokes) - 1, 0) + end_pause

    els = []
    t = 0.0
    common = (
        'fill="none" stroke="#1a1d24" '
        f'stroke-width="{stroke_width:.3f}" '
        'stroke-linecap="round" stroke-linejoin="round"'
    )
    for s, draw in zip(strokes, draw_times):
        start, end = t / total, (t + draw) / total
        t += draw + stroke_gap
        length = _length(s)

        if length < 1e-6:
            # A single-point stroke (a dot): pop it in at its moment via opacity.
            px, py, _ = s[0]
            els.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{stroke_width * 0.6:.3f}" '
                f'fill="#1a1d24" opacity="0">'
                f'<animate attributeName="opacity" dur="{total:.3f}s" '
                f'repeatCount="indefinite" values="0;0;1;1" '
                f'keyTimes="0;{start:.4f};{end:.4f};1" calcMode="linear"/></circle>'
            )
            continue

        d = " ".join(
            f"{'M' if i == 0 else 'L'} {px:.2f} {py:.2f}"
            for i, (px, py, _pen) in enumerate(s)
        )
        # Draw-on via stroke-dashoffset. The dash is `length` long with a gap of
        # `length + 2*stroke_width`, and the hidden offset is `length +
        # stroke_width` -- one stroke-width of margin so that while hidden, NO
        # dash boundary sits on the path. Without that margin the round line-cap
        # of the wrapped dash paints a dot at the stroke's start point, so every
        # not-yet-drawn stroke would show a phantom dot. Offset animates to 0,
        # which lays the dash exactly over [0, length] (round caps at both real
        # ends). Every stroke shares one dur = the full cycle and loops forever,
        # so they stay perfectly in sync; keyTimes hold hidden -> draw -> visible.
        gap = length + 2 * stroke_width
        hidden = length + stroke_width
        els.append(
            f'<path d="{d}" {common} '
            f'stroke-dasharray="{length:.3f} {gap:.3f}" '
            f'stroke-dashoffset="{hidden:.3f}">'
            f'<animate attributeName="stroke-dashoffset" dur="{total:.3f}s" '
            f'repeatCount="indefinite" values="{hidden:.3f};{hidden:.3f};0;0" '
            f'keyTimes="0;{start:.4f};{end:.4f};1" calcMode="linear"/></path>'
        )

    vb = f"{-pad:.2f} {-pad:.2f} {w + 2 * pad:.2f} {h + 2 * pad:.2f}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}">'
        + "".join(els)
        + "</svg>"
    )


def main():
    ap = argparse.ArgumentParser(description="Render strokes to an animated SVG")
    _ = ap.add_argument("input", help="input JSON (generated.json or a recording)")
    _ = ap.add_argument("output", help="output .svg path")
    _ = ap.add_argument(
        "--seconds-per-point",
        type=float,
        default=0.02,
        help="animation time spent per pen point (higher = slower)",
    )
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    svg = strokes_to_animated_svg(
        data["dots"], seconds_per_point=args.seconds_per_point
    )
    with open(args.output, "w") as f:
        f.write(svg)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
