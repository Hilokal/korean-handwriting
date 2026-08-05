"""Fake multi-character samples by stitching real single-character recordings.

This exists to exercise the sliding attention window *before* real line-level
(text, recording) data is available. Single-character data (U == 1) can't test
the window at all -- with one character there is nowhere for kappa to slide. By
concatenating existing glyphs into synthetic 2-3 character "lines" we can watch
the window advance from one character to the next and confirm the multi-character
forward / collate / generate paths are wired correctly.

What this does NOT prove: that the window learns to align to the *right*
character. That needs a training run (kappa's bias makes it drift forward on its
own; only training ties the drift to the actual A->B stroke transition).
"""

import torch
from handwriting_dataset import HandwritingData, HandwritingDataset
from torch.utils.data import Dataset


def concat_samples(
    samples: list[HandwritingData], gap_frac: float = 0.4
) -> HandwritingData:
    """Stitch single-character samples into one multi-character sample.

    Each input is a single-char HandwritingData: strokes (N, 4) of deltas
    [dx, dy, penState, f], tokens (1, 3). Each glyph is reconstructed to absolute
    coordinates, laid out left-to-right (gap between glyphs = `gap_frac` of the
    glyph's own width, so it is scale-invariant -- this data is normalized to
    ~2px glyphs) with baselines aligned, then re-encoded as deltas.

    penState flags are preserved, so each glyph's final point stays end-of-stroke
    (1). Because a point starts a new stroke when the *previous* point is
    end-of-stroke, the between-glyph move renders as a pen-up jump -- the same
    inter-character motion real line data will contain.

    Returns HandwritingData with strokes (M, 4) and tokens (U, 3), U = len(samples).
    """
    # (x, y, penState, f); x/y absolute, f already standardized (absolute too)
    abs_pts: list[tuple[float, float, float, float]] = []
    tokens: list[torch.Tensor] = []
    x_offset = 0.0

    for s in samples:
        strokes = s["strokes"]
        tokens.append(s["tokens"][0])  # (3,) -- single-char tokens are (1, 3)

        cum = torch.cumsum(strokes[:, :2], dim=0)  # (N, 2) positions rel. to glyph
        gx_min = cum[:, 0].min()
        gy_min = cum[:, 1].min()
        xs = cum[:, 0] - gx_min + x_offset  # left edge at x_offset
        ys = cum[:, 1] - gy_min  # baseline at y = 0

        for i in range(strokes.size(0)):
            abs_pts.append(
                (xs[i].item(), ys[i].item(), strokes[i, 2].item(), strokes[i, 3].item())
            )

        width = (cum[:, 0].max() - gx_min).item()
        x_offset += width * (1.0 + gap_frac)

    # Re-encode the laid-out line as deltas (f stays absolute).
    out = []
    px, py = abs_pts[0][0], abs_pts[0][1]
    for x, y, pen, f in abs_pts:
        out.append([x - px, y - py, pen, f])
        px, py = x, y

    return {
        "strokes": torch.tensor(out, dtype=torch.float32),
        "tokens": torch.stack(tokens),  # (U, 3)
    }


class SyntheticMultiCharDataset(Dataset[HandwritingData]):
    """Multi-character samples faked by concatenating real single-char recordings.

    U is randomized in [1, max_chars] so the collate padding and the attention
    token-mask are exercised alongside the window itself.
    """

    def __init__(
        self,
        base: HandwritingDataset | None = None,
        num_samples: int = 600,
        max_chars: int = 3,
        gap_frac: float = 0.4,
        seed: int = 0,
    ):
        base = base or HandwritingDataset()
        usable = [i for i in range(len(base)) if base[i]["strokes"].size(0) > 1]
        gen = torch.Generator().manual_seed(seed)

        self.items: list[HandwritingData] = []
        for _ in range(num_samples):
            u = int(torch.randint(1, max_chars + 1, (1,), generator=gen).item())
            picks = [
                usable[int(torch.randint(len(usable), (1,), generator=gen).item())]
                for _ in range(u)
            ]
            self.items.append(concat_samples([base[i] for i in picks], gap_frac))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> HandwritingData:
        return self.items[idx]
