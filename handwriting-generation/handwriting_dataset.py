import json
import os
import statistics
from pathlib import Path
from typing import TypedDict

import torch
from tokenizer import tokenize
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent

characters = ["간", "안", "아"]


class HandwritingData(TypedDict):
    strokes: torch.Tensor
    tokens: torch.Tensor  # (U, 4) -- [leading, vowel, trailing, symbol] per unit


class HandwritingDataset(Dataset[HandwritingData]):
    files: list[dict[str, str | int]]

    def __init__(self):
        root_dir = HERE / ".." / "data"

        self.files = []

        for character in characters:
            subdir = root_dir / character

            for f in os.listdir(subdir):
                if f.endswith(".json"):
                    self.files.append(
                        {"filename": os.path.join(subdir, f), "character": character}
                    )

        self.files.sort(key=lambda x: x["filename"])

        # Pre-load + transform every sample once. The dataset fits easily in RAM,
        # and this avoids re-reading + re-parsing + re-transforming each file on
        # every epoch (which dominated training time for this tiny model).
        self.items: list[HandwritingData] = []
        for item in self.files:
            with open(str(item["filename"]), "r") as fp:
                strokes = self.transform(json.load(fp))

            # tokens is (U, 3): one jamo triple per character. U == 1 for the
            # current single-character folders; the attention window handles
            # U > 1 once line-level (text, recording) data is available.
            tokens = tokenize(str(item["character"]))
            self.items.append({"strokes": strokes, "tokens": tokens})

    def __len__(self):
        return len(self.files)

    @staticmethod
    def transform(
        data: dict, max_delta: float = 10.0, y_outlier: float = 3.0
    ) -> torch.Tensor:
        """Transform JSON data into a tensor of shape (N, 3) with [dx, dy, penState].

        Uses delta-based coordinates: each point is the change from the previous point.
        penState is a **binary end-of-stroke** flag (Graves-style trailing edge):
          0 = mid-stroke
          1 = last point of a stroke (pen lifts *after* it) — fires on the final
              stroke too, since it's also a stroke end.
        There is no separate end-of-sequence class: termination is handled at
        generation time by stopping after the character's expected stroke count
        (see model.generate `num_strokes`). Anchoring the lift on the stroke's
        *last* point means the next step's input carries a "a stroke just ended"
        cue, letting the model condition the upcoming jump offset on it.

        Points that would create deltas larger than max_delta are skipped (likely glitches).
        """
        PEN_MOVE = 1

        # Off-line spike repair: a stray sensor point that sits far above/below the
        # line but is reached by a small (< max_delta) delta escapes the jump filter
        # below, yet it stretches the recording into a long thin spike (a single such
        # point, not a drift). Real lines are ~2.5 tall, so drop pen-move points more
        # than y_outlier from the median line height -- keeping the rest of the line
        # rather than discarding the whole recording.
        move_ys = [d["y"] for d in data["dots"] if d["dotType"] == PEN_MOVE]
        med_y = statistics.median(move_ys) if move_ys else 0.0

        # First pass: extract absolute coordinates, tagging each point with a
        # stroke id (incremented at every stroke boundary).
        abs_dots = []
        stroke_id = 0
        new_stroke = True

        for dot in data["dots"]:
            if dot["dotType"] == PEN_MOVE:
                if abs(dot["y"] - med_y) > y_outlier:
                    continue  # off-line spike: drop the point, keep the stroke
                if new_stroke and abs_dots:
                    stroke_id += 1
                abs_dots.append([dot["x"], dot["y"], stroke_id])
                new_stroke = False
            else:
                new_stroke = True

        if not abs_dots:
            return torch.zeros((0, 3), dtype=torch.float32)

        # Second pass: convert to deltas, filtering out large jumps. Keep stroke id.
        dots = []  # [dx, dy, stroke_id]
        prev_x, prev_y = abs_dots[0][0], abs_dots[0][1]

        for x, y, sid in abs_dots:
            dx = x - prev_x
            dy = y - prev_y

            # Skip glitchy points with large deltas
            if abs(dx) > max_delta or abs(dy) > max_delta:
                continue

            dots.append([dx, dy, sid])
            prev_x, prev_y = x, y

        if not dots:
            return torch.zeros((0, 3), dtype=torch.float32)

        # Third pass: derive the binary end-of-stroke flag from stroke ids
        # (robust to the filtering above). A point is end-of-stroke (1) if it's
        # the last *kept* point of its stroke, including the final stroke.
        out = []
        for i, (dx, dy, sid) in enumerate(dots):
            is_last = i == len(dots) - 1
            end_of_stroke = is_last or dots[i + 1][2] != sid
            out.append([dx, dy, 1 if end_of_stroke else 0])

        return torch.tensor(out, dtype=torch.float32)

    def __getitem__(self, idx: int) -> HandwritingData:
        return self.items[idx]


def modal_stroke_counts() -> dict[str, int]:
    """Most common stroke count per character across the dataset.

    Used as the `num_strokes` decode constraint to terminate generation. Stroke
    count = number of end-of-stroke (penState==1) markers.
    """
    from collections import Counter

    ds = HandwritingDataset()
    per_char: dict[str, Counter] = {}
    for i in range(len(ds)):
        ch = str(ds.files[i]["character"])
        ps = ds[i]["strokes"][:, 2].tolist()
        n = sum(1 for p in ps if p == 1)
        per_char.setdefault(ch, Counter())[n] += 1

    return {ch: c.most_common(1)[0][0] for ch, c in per_char.items()}


class ExportDataset(Dataset[HandwritingData]):
    """Loads the production export at ../data/export/recordings/**/*.json.

    Each recording carries its dots (-> strokes via transform) and the line it
    represents in metadata.text (-> tokens via tokenize). Unlike the single-char
    folder loader, the transcript travels inside the file, so this handles real
    multi-character lines (including spaces and punctuation as symbol tokens).

    `meta[i]` keeps each item's text and sentence_id so the split can be done by
    *text* (no sentence shared between train and val), which is what tests whether
    the model generalizes composition rather than memorizing specific lines.
    """

    def __init__(self, export_dir: str | Path | None = None, min_points: int = 2):
        root = (
            Path(export_dir)
            if export_dir
            else HERE / ".." / "data" / "export" / "recordings"
        )

        self.items: list[HandwritingData] = []
        self.meta: list[dict] = []
        for f in sorted(root.glob("**/*.json")):
            with open(f, "r") as fp:
                data = json.load(fp)
            # transform() repairs off-line sensor spikes (drops the stray points,
            # keeps the line), so no whole recording needs discarding for them.
            strokes = HandwritingDataset.transform(data)
            if strokes.size(0) <= min_points:
                continue
            text = data.get("metadata", {}).get("text", "")
            if not text:
                continue
            tokens = tokenize(text)
            if tokens.size(0) == 0:
                continue
            self.items.append({"strokes": strokes, "tokens": tokens})
            self.meta.append(
                {"text": text, "sentence_id": data["metadata"].get("sentenceId")}
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> HandwritingData:
        return self.items[idx]


def split_by_text(
    dataset: ExportDataset, val_frac: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Partition indices so no sentence appears in both train and val.

    Splitting by recording would let the model see a sentence in training and be
    scored on a different recording of the *same* text -- which measures
    reproduction, not generalization. Grouping by sentence_id and holding out
    whole sentences tests whether the window generalizes to unseen text.
    """
    from collections import defaultdict

    groups: dict[object, list[int]] = defaultdict(list)
    for i, m in enumerate(dataset.meta):
        groups[m["sentence_id"]].append(i)

    ids = sorted(groups, key=lambda k: (k is None, k))
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ids), generator=gen).tolist()

    n_val = max(1, int(round(len(ids) * val_frac)))
    val_ids = {ids[order[j]] for j in range(n_val)}

    train_idx, val_idx = [], []
    for sid, members in groups.items():
        (val_idx if sid in val_ids else train_idx).extend(members)
    return train_idx, val_idx
