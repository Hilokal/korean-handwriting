#!/usr/bin/env python3
"""Build the public Hugging Face dataset from the production export.

Reads ../data/export/ (manifest.json, sentences.jsonl, recordings/**.json --
produced by download_export.sh) and writes a publishable dataset directory:

    data/recordings.parquet   one row per recording, strokes nested per row
    data/sentences.parquet    the sentence pool with provenance
    README.md                 dataset card (YAML front matter + description)

Identifying fields in the raw export (writer name, pen MAC, Ncode page ids)
are NOT carried over. Only PEN_MOVE dots are kept -- DOWN/UP/HOVER events
carry no trustworthy coordinates (see collection-app strokeSvg.ts) and the
training loader ignores them too; stroke boundaries become the nesting.
Points are published raw: the loader's spike/jump repair is documented in the
card so consumers can reproduce or improve on it.

Usage:
    ./venv/bin/python export_hf_dataset.py                 # build to ../data/hf-dataset
    ./venv/bin/python export_hf_dataset.py --push USER/REPO  # build, then upload
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
EXPORT_DIR = HERE / ".." / "data" / "export"
OUT_DIR = HERE / ".." / "data" / "hf-dataset"

PEN_MOVE = 1
WRITER_ID = 1  # the single writer behind every published recording (export userId 2)
WRITER_CREDIT = "Jo YW"  # credited by name at the writer's request
GITHUB_URL = "https://github.com/Hilokal/korean-handwriting"
DEMO_URL = "https://hangul.ink"

STROKE_TYPE = pa.struct(
    [
        ("x", pa.list_(pa.float32())),
        ("y", pa.list_(pa.float32())),
        ("force", pa.list_(pa.int16())),
        ("t_ms", pa.list_(pa.int32())),
        ("tilt_x", pa.list_(pa.int16())),
        ("tilt_y", pa.list_(pa.int16())),
        ("twist", pa.list_(pa.int16())),
    ]
)

RECORDING_SCHEMA = pa.schema(
    [
        ("recording_id", pa.int32()),
        ("sentence_id", pa.int32()),
        ("writer_id", pa.int8()),
        ("text", pa.string()),
        ("sentence_text", pa.string()),
        ("chunk_index", pa.int8()),
        ("text_source", pa.string()),
        ("source", pa.string()),
        ("recorded_at", pa.string()),
        ("duration_ms", pa.int32()),
        ("num_strokes", pa.int16()),
        ("num_points", pa.int32()),
        ("strokes", pa.list_(STROKE_TYPE)),
    ]
)

SENTENCE_SCHEMA = pa.schema(
    [
        ("sentence_id", pa.int32()),
        ("text", pa.string()),
        ("text_source", pa.string()),
        ("source", pa.string()),
        ("num_recordings", pa.int16()),
    ]
)


def text_source(source: str) -> str:
    return "synthetic" if source.startswith("synthetic") else "wikipedia"


def strokes_from_dots(dots: list[dict]) -> tuple[list[dict], int]:
    """Group PEN_MOVE dots into strokes; a non-move dot ends the current stroke.

    Returns (strokes, duration_ms). Timestamps are made relative to the first
    move dot so rows don't carry wall-clock time.
    """
    strokes: list[dict] = []
    cur: dict | None = None
    t0: int | None = None
    t_last = 0
    for d in dots:
        if d["dotType"] != PEN_MOVE:
            cur = None
            continue
        if t0 is None:
            t0 = d["timeStamp"]
        if cur is None:
            cur = {k: [] for k in ("x", "y", "force", "t_ms", "tilt_x", "tilt_y", "twist")}
            strokes.append(cur)
        ang = d.get("angle") or {}
        t_last = d["timeStamp"] - t0
        cur["x"].append(float(d["x"]))
        cur["y"].append(float(d["y"]))
        cur["force"].append(int(d.get("f", 0)))
        cur["t_ms"].append(int(t_last))
        cur["tilt_x"].append(int(ang.get("tx", 0)))
        cur["tilt_y"].append(int(ang.get("ty", 0)))
        cur["twist"].append(int(ang.get("twist", 0)))
    return strokes, t_last


def build(export_dir: Path, out_dir: Path) -> dict:
    sentences = {}
    with open(export_dir / "sentences.jsonl") as fp:
        for line in fp:
            s = json.loads(line)
            sentences[s["id"]] = s
    with open(export_dir / "manifest.json") as fp:
        manifest = json.load(fp)

    rows = []
    per_sentence: Counter[int] = Counter()
    for m in sorted(manifest, key=lambda r: r["recordingId"]):
        with open(export_dir / m["file"]) as fp:
            rec = json.load(fp)
        strokes, duration = strokes_from_dots(rec["dots"])
        if not strokes:
            continue
        sent = sentences[m["sentenceId"]]
        per_sentence[m["sentenceId"]] += 1
        rows.append(
            {
                "recording_id": m["recordingId"],
                "sentence_id": m["sentenceId"],
                "writer_id": WRITER_ID,
                "text": m["text"],
                "sentence_text": m["sentenceText"],
                "chunk_index": m["chunkIndex"],
                "text_source": text_source(sent["source"]),
                "source": sent["source"],
                "recorded_at": m["createdAt"].replace(" ", "T") + "Z",
                "duration_ms": duration,
                "num_strokes": len(strokes),
                "num_points": sum(len(s["x"]) for s in strokes),
                "strokes": strokes,
            }
        )

    sent_rows = [
        {
            "sentence_id": sid,
            "text": s["text"],
            "text_source": text_source(s["source"]),
            "source": s["source"],
            "num_recordings": per_sentence.get(sid, 0),
        }
        for sid, s in sorted(sentences.items())
        if per_sentence.get(sid, 0) > 0
    ]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data").mkdir(parents=True)
    rec_table = pa.Table.from_pylist(rows, schema=RECORDING_SCHEMA)
    pq.write_table(rec_table, out_dir / "data" / "recordings.parquet", compression="zstd")
    sent_table = pa.Table.from_pylist(sent_rows, schema=SENTENCE_SCHEMA)
    pq.write_table(sent_table, out_dir / "data" / "sentences.parquet", compression="zstd")

    stats = {
        "recordings": len(rows),
        "sentences": len(sent_rows),
        "strokes": sum(r["num_strokes"] for r in rows),
        "points": sum(r["num_points"] for r in rows),
        "hours": sum(r["duration_ms"] for r in rows) / 3_600_000,
        "by_source": Counter(r["text_source"] for r in rows),
        "sentences_by_source": Counter(r["text_source"] for r in sent_rows),
        "batches": sorted({r["source"] for r in sent_rows if r["text_source"] == "synthetic"}),
    }
    (out_dir / "README.md").write_text(dataset_card(stats))
    return stats


def dataset_card(st: dict) -> str:
    batches = "\n".join(f"  - `{b}`" for b in st["batches"])
    return f"""---
license: cc-by-sa-4.0
language:
  - ko
pretty_name: Korean Handwriting Strokes (hangul.ink)
tags:
  - handwriting
  - online-handwriting
  - hangul
  - korean
  - pen-strokes
  - handwriting-synthesis
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files: data/recordings.parquet
  - config_name: sentences
    data_files: data/sentences.parquet
---

# Korean Handwriting Strokes

{st["recordings"]:,} lines of Korean handwriting captured as **pen strokes** (not
images) with a Neo Smartpen: x/y position, pen-tip force, timing, and pen tilt
for every sample, grouped into strokes. This is the training data behind
[hangul.ink]({DEMO_URL}), a browser-side recurrent network that writes Hangul by
hand, released for Hangul Day 2026. Code for collection, training, and the demo
is at [{GITHUB_URL.removeprefix("https://")}]({GITHUB_URL}).

| | |
|---|---|
| Recordings (lines) | {st["recordings"]:,} |
| Distinct sentences | {st["sentences"]:,} |
| Strokes | {st["strokes"]:,} |
| Pen samples | {st["points"]:,} |
| Recording time | {st["hours"]:.1f} h |
| Writers | 1 |

All recordings are by a single writer, **{WRITER_CREDIT}**, a native Korean
speaker who wrote every line for this project with a Neo Smartpen on Ncode paper. The pen streams at roughly
50 samples per second.

## Format

`data/recordings.parquet` has one row per recording. Long sentences were
written as several lines; `chunk_index` orders the chunks of one
`sentence_text`, and `text` is what this particular recording says.

| column | type | meaning |
|---|---|---|
| `recording_id` | int | unique id |
| `sentence_id` | int | joins to `sentences` |
| `writer_id` | int | always 1 in this release |
| `text` | string | the transcript of this line |
| `sentence_text` | string | the full sentence the line belongs to |
| `chunk_index` | int | 0-based position of this line within the sentence |
| `text_source` | string | `wikipedia` or `synthetic` (see Licensing) |
| `source` | string | Korean Wikipedia article title, or the synthetic batch name |
| `recorded_at` | string | UTC timestamp of the recording |
| `duration_ms` | int | time from first to last pen sample |
| `num_strokes`, `num_points` | int | sizes of `strokes` |
| `strokes` | list of struct | one struct per pen-down..pen-up stroke, each holding parallel arrays `x`, `y` (Ncode paper coordinates, float), `force` (pen-tip pressure, raw 0..~1000), `t_ms` (ms since the recording's first sample), `tilt_x`, `tilt_y`, `twist` (pen orientation, degrees) |

`x`/`y` are absolute positions on the page in Ncode units (the paper's dot
grid, about 2.4 mm per unit). A written line is roughly 2.5 units tall.
Recordings are not registered to a common origin; models typically consume
deltas between consecutive points.

`data/sentences.parquet` lists each sentence once with its provenance and how
many recordings exist for it.

```python
from datasets import load_dataset

ds = load_dataset("USER/REPO", split="train")
row = ds[0]
print(row["text"], row["num_strokes"])
first = row["strokes"][0]
xy = list(zip(first["x"], first["y"]))
```

## Known noise

Points are published exactly as the pen reported them. Two artefacts are
worth handling:

- **Ncode misreads**: occasionally a single move sample lands far off the
  line (a spike). The training loader drops move points more than a few units
  from the line's median y, and skips any step whose delta is implausibly large.
- **Early recordings without force**: all rows in this release carry force.

The reference loader (`handwriting_dataset.py` in the repository) shows the
exact cleaning used to train the released model.

## How the text was chosen

The first {st["sentences_by_source"]["wikipedia"]:,} sentences are from Korean Wikipedia (dictionary-form prose).
A model trained on those alone had holes: for example, it had almost never
seen the polite spoken ending 요 at the end of a line. The remaining sentences
were generated synthetically with Claude in targeted batches to fill such
gaps:

{batches}

## Licensing and attribution

The dataset is released under **CC BY-SA 4.0**.

- Sentences with `text_source = wikipedia` are quoted from the Korean
  Wikipedia article named in `source`, which is itself
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Credit
  Korean Wikipedia and the article when redistributing those transcripts.
- Sentences with `text_source = synthetic` and all stroke data are original to
  this project.

Please attribute as: *Korean Handwriting Strokes by Jonathan Baudanza
(hangul.ink), handwriting by {WRITER_CREDIT}, CC BY-SA 4.0; transcripts partly
from Korean Wikipedia (CC BY-SA 4.0).*

Use for education, art, research, and products is welcome. {WRITER_CREDIT}
agreed to this public release and to being credited by name.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--push", metavar="USER/REPO", help="upload the built directory to this Hub dataset repo")
    args = ap.parse_args()

    st = build(args.export_dir.resolve(), args.out_dir.resolve())
    print(f"recordings {st['recordings']:,}  sentences {st['sentences']:,}  "
          f"strokes {st['strokes']:,}  points {st['points']:,}  {st['hours']:.1f} h")
    print("recordings by text source:", dict(st["by_source"]))
    print("sentences by text source:", dict(st["sentences_by_source"]))
    for f in sorted(args.out_dir.resolve().rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(args.out_dir.resolve())}  {f.stat().st_size / 1e6:.1f} MB")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, repo_type="dataset", exist_ok=True)
        card = (args.out_dir / "README.md")
        card.write_text(card.read_text().replace("USER/REPO", args.push))
        api.upload_folder(
            folder_path=str(args.out_dir),
            repo_id=args.push,
            repo_type="dataset",
            commit_message="Hangul Day 2026 release",
        )
        print(f"pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
