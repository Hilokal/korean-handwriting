#!/usr/bin/env python3
"""Parity check for the ONNX single-step export.

Two comparisons, over a multi-step rollout with realistic inputs:

  1. SingleStep wrapper (PyTorch) vs HandwritingRNN.forward threading state --
     catches the wrapper drifting from model.py.
  2. ONNX Runtime session vs the wrapper -- catches export bugs.

Each implementation carries its OWN recurrent state across steps, so numerical
drift compounds the way it would in production; the tolerance below reflects
~200 steps of fp32 accumulation.

    python test_onnx_parity.py --model best_model.emb8.pt --num-layers 3 \
        --embedding-size 8 --onnx ../demo-app/public/model/handwriting-step.onnx
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch

from export_onnx import SingleStep, load_model
from tokenizer import tokenize

STEPS = 200
TOL = 1e-3
TEXT = "안녕하세요, 반갑습니다!"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_model.eot.pt")
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--embedding-size", type=int, default=8)
    ap.add_argument("--onnx", default="../demo-app/public/model/handwriting-step.onnx")
    ap.add_argument("--onehot", action="store_true", default=bool(os.environ.get("ONEHOT")))
    args = ap.parse_args()

    torch.manual_seed(0)
    model = load_model(args)
    step = SingleStep(model)
    step.eval()
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    tokens = tokenize(TEXT).unsqueeze(0)  # (1, U, 4)
    U = tokens.size(1)
    # U+1 indices: the last is the phantom past-the-end position (phi column U)
    u = torch.arange(U + 1, dtype=torch.float32).view(1, 1, U + 1)
    mask = torch.ones(1, U)
    L, H = args.num_layers, args.hidden_size
    W, K = model.gru.window_dim, model.gru.sliding_window_k

    # Reference (model.forward) state
    ref_state = None
    # Wrapper state
    wh = torch.zeros(L, 1, H)
    ww = torch.zeros(1, W)
    wk = torch.zeros(1, K)
    wp = torch.zeros(1, 2)
    # ONNX state
    oh, ow_, ok_, op_ = (t.numpy().copy() for t in (wh, ww, wk, wp))

    # Realistic-magnitude random pen inputs [dx, dy, penState, f] (deltas are
    # ~unit scale; f is standardized force, also ~unit scale).
    xs = torch.randn(STEPS, 4) * 0.7
    xs[:, 2] = (torch.rand(STEPS) < 0.08).float()  # occasional pen lifts
    xs[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])

    max_wrapper = max_onnx = 0.0
    with torch.no_grad():
        for t in range(STEPS):
            x = xs[t : t + 1]  # (1, 4)

            mdn_ref, pen_ref, ref_state, _phi = model(
                tokens, x.unsqueeze(1), mask, ref_state
            )
            mdn_w, pen_w, wh, ww, wk, wp, _ = step(
                tokens, x, u, mask, wh, ww, wk, wp
            )
            d = max(
                (mdn_ref[:, -1] - mdn_w).abs().max().item(),
                (pen_ref[:, -1] - pen_w).abs().max().item(),
            )
            max_wrapper = max(max_wrapper, d)

            outs = sess.run(
                None,
                {
                    "tokens": tokens.numpy(),
                    "x": x.numpy(),
                    "u": u.numpy(),
                    "mask": mask.numpy(),
                    "hidden": oh,
                    "w": ow_,
                    "kappa": ok_,
                    "pos": op_,
                },
            )
            mdn_o, pen_o, oh, ow_, ok_, op_, _phi_o = outs
            d = max(
                np.abs(mdn_o - mdn_w.numpy()).max(),
                np.abs(pen_o - pen_w.numpy()).max(),
            )
            max_onnx = max(max_onnx, float(d))

    print(f"steps: {STEPS}, text: {TEXT!r} (U={U})")
    print(f"max |wrapper - model.forward| = {max_wrapper:.2e}")
    print(f"max |onnx - wrapper|          = {max_onnx:.2e}")
    assert max_wrapper < TOL, "wrapper drifted from model.py"
    assert max_onnx < TOL, "ONNX export mismatch"
    print("PARITY OK")


if __name__ == "__main__":
    main()
