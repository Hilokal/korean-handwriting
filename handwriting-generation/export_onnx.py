#!/usr/bin/env python3
"""Export the model as a single-step ONNX graph for in-browser inference.

The autoregressive loop cannot be exported wholesale -- MDN sampling happens
between steps -- so this exports ONE step of the recurrence:

    (tokens, x, u, mask, hidden, w, kappa, pos)
        -> (mdn_raw, pen_logit, hidden_out, w_out, kappa_out, pos_out, phi)

and the loop (sampling, termination) lives in the browser (demo-app).

Why a wrapper instead of exporting HandwritingRNN.forward directly: the tracer
bakes `torch.arange(U)` / `torch.ones(B, U)` into constants, which would pin the
graph to one text length. The wrapper takes `u` (character indices 0..U-1) and
`mask` as graph *inputs* -- trivial for the JS caller to supply -- and re-derives
the step from the model's own submodules. test_onnx_parity.py asserts this
wrapper matches model.forward exactly, so it cannot silently drift.

Usage (matches the bundled checkpoint; needs `pip install onnx onnxscript`,
and `onnxruntime` for the parity test):
    python export_onnx.py --model best_model.emb8.pt --num-layers 3 \
        --embedding-size 8 --out-dir ../demo-app/public/model
    python test_onnx_parity.py --model best_model.emb8.pt --num-layers 3 \
        --embedding-size 8    # must print PARITY OK before shipping
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from handwriting_dataset import FORCE_MEAN, FORCE_STD
from model import HandwritingRNN
from tokenizer import EOT_SYMBOL_ID


class SingleStep(nn.Module):
    """One timestep of HandwritingRNN.generate(), trace-friendly.

    Shapes (B=1 always; U = number of text units, dynamic):
        tokens: (1, U, 4) int64   x:     (1, 4) float   u:    (1, 1, U+1) float
        mask:   (1, U)    float   hidden:(L, 1, H)      w:    (1, W)
        kappa:  (1, K)            pos:   (1, 2)
    x is [dx, dy, penState, f] (f = absolute standardized pen force).
    u is the character-index grid *including the phantom past-the-end position*
    (0..U inclusive, U+1 entries), matching GRUWithSlidingAttention: the phi
    output's last column is the phantom weight used for Graves' termination
    test (stop when phi[U] exceeds every real character's phi).
    """

    def __init__(self, m: HandwritingRNN):
        super().__init__()
        self.m = m

    def forward(self, tokens, x, u, mask, hidden, w, kappa, pos):
        m = self.m
        gru = m.gru

        # Absolute position: single step, so cumsum reduces to one addition.
        abs_pos = pos + x[:, :2]  # (1, 2)
        norm_pos = (abs_pos - m.pos_mean.view(1, 2)) / m.pos_std.view(1, 2)
        aug_x = torch.cat((x, norm_pos[:, : m.abs_pos_dim]), dim=-1)  # (1, 4+P)

        # Per-unit conditioning vectors (jamo concat, or symbol embedding).
        leading = m.leading_embeddings(tokens[:, :, 0])
        vowel = m.vowel_embeddings(tokens[:, :, 1])
        trailing = m.trailing_embeddings(tokens[:, :, 2])
        c_jamo = torch.cat((leading, vowel, trailing), dim=-1)  # (1, U, W)
        symbol = tokens[:, :, 3]
        c_symbol = m.symbol_embeddings(symbol)
        c = torch.where((symbol > 0).unsqueeze(-1), c_symbol, c_jamo)

        # Layer 0 sees the previous step's window w_{t-1}.
        h = gru.layers[0](torch.cat([aug_x, w], dim=-1), hidden[0])
        new_hidden = [h]

        # Attention window from the first hidden layer (Graves eqs. 46-51).
        alpha = torch.exp(gru.alpha_head(h)).unsqueeze(-1)  # (1, K, 1)
        beta = torch.exp(gru.beta_head(h)).unsqueeze(-1)  # (1, K, 1)
        kappa_out = kappa + torch.exp(gru.kappa_head(h))  # (1, K)
        phi = (alpha * torch.exp(-beta * (kappa_out.unsqueeze(-1) - u) ** 2)).sum(1)
        # (1, U+1): last column is the unmasked phantom past-the-end weight.
        phi_real = phi[:, :-1] * mask  # (1, U)
        w_out = torch.bmm(phi_real.unsqueeze(1), c).squeeze(1)  # (1, W)
        phi = torch.cat([phi_real, phi[:, -1:]], dim=-1)  # masked + phantom

        # Higher layers: input skip + layer below + current window.
        below = h
        for i in range(1, gru.num_layers):
            layer_in = torch.cat([aug_x, below, w_out], dim=-1)
            h = gru.layers[i](layer_in, hidden[i])
            new_hidden.append(h)
            below = h

        out = torch.cat(new_hidden, dim=-1)  # (1, H*L) -- output skip
        mdn_raw = m.mdn_head(out)  # (1, 8*num_mixtures)
        pen_logit = m.pen_head(out)  # (1, 1)
        hidden_out = torch.stack(new_hidden, dim=0)  # (L, 1, H)

        return mdn_raw, pen_logit, hidden_out, w_out, kappa_out, abs_pos, phi


def load_model(args) -> HandwritingRNN:
    model = HandwritingRNN(
        input_size=4,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.0,
        embedding_size=args.embedding_size,
        onehot=args.onehot,
    )
    model.load_state_dict(
        torch.load(args.model, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def example_inputs(model: HandwritingRNN, num_layers: int, U: int = 5):
    hidden_size = model.gru.hidden_size
    return (
        torch.randint(0, 5, (1, U, 4), dtype=torch.int64),
        torch.zeros(1, 4),
        torch.arange(U + 1, dtype=torch.float32).view(1, 1, U + 1),
        torch.ones(1, U),
        torch.zeros(num_layers, 1, hidden_size),
        torch.zeros(1, model.gru.window_dim),
        torch.zeros(1, model.gru.sliding_window_k),
        torch.zeros(1, 2),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="best_model.eot.pt")
    ap.add_argument("--hidden-size", type=int, default=int(os.environ.get("HIDDEN_SIZE", "128")))
    ap.add_argument("--num-layers", type=int, default=int(os.environ.get("NUM_LAYERS", "3")))
    ap.add_argument("--embedding-size", type=int, default=int(os.environ.get("EMBEDDING_SIZE", "8")))
    ap.add_argument("--onehot", action="store_true", default=bool(os.environ.get("ONEHOT")))
    ap.add_argument("--out-dir", default="../demo-app/public/model")
    args = ap.parse_args()

    model = load_model(args)
    step = SingleStep(model)
    step.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "handwriting-step.onnx"

    input_names = ["tokens", "x", "u", "mask", "hidden", "w", "kappa", "pos"]
    output_names = ["mdn_raw", "pen_logit", "hidden_out", "w_out", "kappa_out", "pos_out", "phi"]
    with torch.no_grad():
        torch.onnx.export(
            step,
            example_inputs(model, args.num_layers),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "tokens": {1: "U"},
                "u": {2: "U1"},  # U+1 entries: real indices + phantom
                "mask": {1: "U"},
                "phi": {1: "U1"},
            },
            opset_version=17,
            # The dynamo exporter trips over the dynamic-U GRUCell graph; the
            # legacy tracer handles it fine (u/mask are inputs by design).
            dynamo=False,
        )

    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:12]
    meta = {
        "version": digest,
        "checkpoint": os.path.basename(args.model),
        "hiddenSize": args.hidden_size,
        "numLayers": args.num_layers,
        "embeddingSize": args.embedding_size,
        "numMixtures": model.num_mixtures,
        "windowDim": model.gru.window_dim,
        "slidingWindowK": model.gru.sliding_window_k,
        "absPosDim": model.abs_pos_dim,
        # x is [dx, dy, penState, f]; f is standardized by these constants
        # (de-standardize sampled pressure with f * forceStd + forceMean).
        "inputSize": 4,
        "forceMean": FORCE_MEAN,
        "forceStd": FORCE_STD,
        # The u input takes U+1 indices (0..U); phi's last column is the
        # phantom past-the-end weight for Graves' termination test.
        "phantomPhi": True,
        # The tokenizer appends an EOT unit (this symbol id) after the last
        # real character; generator.ts must do the same, and should stop at a
        # stroke boundary once argmax(phi[:U]) reaches the EOT unit (phantom
        # phi test as backstop). Checkpoints trained before EOT don't have it.
        "eotSymbolId": EOT_SYMBOL_ID,
    }
    (out_dir / "model-meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    size_kb = onnx_path.stat().st_size / 1024
    print(f"wrote {onnx_path} ({size_kb:.0f} KB), version {digest}")
    print(f"wrote {out_dir / 'model-meta.json'}")


if __name__ == "__main__":
    main()
