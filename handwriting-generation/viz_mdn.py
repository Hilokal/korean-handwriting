#!/usr/bin/env python3
"""Dump a per-step MDN trace for the mixture-density visualization.

Runs one generation with model.generate(trace=...) and writes a JSON file
holding the strokes plus, for every generated point, the mixture the point was
sampled from -- each component reduced to its (dx, dy) marginal drawn as a
1-sigma ellipse: weight, center offset, radii, and rotation.

    NUM_LAYERS=3 EMBEDDING_SIZE=8 python viz_mdn.py --text "아름다운 문자입니다" --seed 42

Optionally bakes the trace into a self-contained HTML page (--html) by
substituting __TRACE_JSON__ in viz_mdn_template.html.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from handwriting_dataset import FORCE_MEAN, FORCE_STD
from inference import get_device, load_model

# Components below this mixture weight are dropped from the dump -- invisible
# at any reasonable opacity mapping, and they dominate the file size otherwise.
PI_FLOOR = 0.002


def ellipse_params(sx: float, sy: float, rho: float):
    """1-sigma ellipse of the (dx, dy) marginal: (r_major, r_minor, angle_rad).

    Eigen-decomposition of the 2x2 covariance [[sx^2, rho*sx*sy],
    [rho*sx*sy, sy^2]]; the angle is the major axis' rotation from +x.
    """
    a, b, c = sx * sx, rho * sx * sy, sy * sy
    half_tr = (a + c) / 2
    d = math.sqrt(((a - c) / 2) ** 2 + b * b)
    lam1, lam2 = half_tr + d, max(half_tr - d, 0.0)
    angle = 0.5 * math.atan2(2 * b, a - c)
    return math.sqrt(lam1), math.sqrt(lam2), angle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--model", default="best_model.eot.pt")
    _ = parser.add_argument("--text", default="아름다운 문자입니다")
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument("--bias", type=float, default=0.0)
    _ = parser.add_argument("--temperature", type=float, default=1.0)
    _ = parser.add_argument("--max-len", type=int, default=800)
    _ = parser.add_argument("--output", default="mdn_trace.json")
    _ = parser.add_argument(
        "--html",
        default=None,
        help="Also write a self-contained HTML page (fills viz_mdn_template.html)",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    model = load_model(args.model, device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    start_seq = torch.zeros(1, 1, 4, device=device)
    trace: list[dict] = []
    generated = model.generate(
        start_seq,
        character=args.text,
        max_len=args.max_len,
        temperature=args.temperature,
        bias=args.bias,
        device=device,
        trace=trace,
    )
    seq = generated.squeeze(0).cpu().numpy()

    # Trace entry i belongs to generated point start_len + i; the parked-pen
    # trim can drop trailing points, so cut the trace to match.
    start_len = 1
    trace = trace[: len(seq) - start_len]
    print(f"{len(seq)} points ({len(trace)} generated + {start_len} seed)")

    dots = [
        {
            "x": round(float(p[0]), 4),
            "y": round(float(p[1]), 4),
            "penState": int(p[2]),
            "f": round(float(p[3]) * FORCE_STD + FORCE_MEAN, 1),
        }
        for p in seq
    ]

    steps = []
    for entry in trace:
        comps = []
        for j, (pi, mu, sigma, rho) in enumerate(
            zip(entry["pi"], entry["mu"], entry["sigma"], entry["rho"])
        ):
            if pi < PI_FLOOR and j != entry["k"]:
                continue
            r1, r2, angle = ellipse_params(sigma[0], sigma[1], rho)
            comps.append(
                {
                    "pi": round(pi, 4),
                    "mx": round(mu[0], 4),
                    "my": round(mu[1], 4),
                    "r1": round(r1, 4),
                    "r2": round(r2, 4),
                    "a": round(angle, 4),
                    "sampled": j == entry["k"],
                }
            )
        comps.sort(key=lambda c: -c["pi"])
        steps.append(
            {
                "penP": round(entry["pen_p"], 4),
                # (U+1,) attention over units: one per character, then the EOT
                # unit, then the phantom past-the-end column
                "phi": [round(p, 4) for p in entry["phi"]],
                "comps": comps,
            }
        )

    out = {
        "text": args.text,
        "seed": args.seed,
        "bias": args.bias,
        "temperature": args.temperature,
        "model": args.model,
        "startLen": start_len,
        "dots": dots,
        "steps": steps,
    }
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.output}")

    if args.html:
        template = Path(__file__).with_name("viz_mdn_template.html").read_text()
        html = template.replace("__TRACE_JSON__", json.dumps(out))
        Path(args.html).write_text(html)
        print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
