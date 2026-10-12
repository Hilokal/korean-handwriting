#!/usr/bin/env python3
"""
Build a curated batch of short lines that exercise LINE ENDINGS: single words
and short phrases, half of them with no terminal punctuation.

Why: every sentence source in the pipeline (Wikipedia splitter, the LLM
generator's "must end in . ! ?" rule) produces punctuation-terminated lines --
2,562 of 2,594 corpus sentences, 0 ending in a bare 요, 2 in a bare 다. The
attention window has therefore never seen [syllable][EOT], and at generation
time a bare ending (안녕하세요, 좋다, 조슈아) terminates one stroke early: the
ㅏ tick or ㅛ bar is lost and 다 reads as 디 (HN launch feedback, 2026-10-11).

The dropped stroke is always the *last stroke of the final syllable*, so lines
are chosen by the shape of that stroke, not by vocabulary:
  tick      -- short final tick: ㅏ ㅑ ㅜ ㅠ ㅘ          (the most fragile class)
  bar       -- long horizontal:  ㅗ ㅛ ㅡ
  vertical  -- long vertical:    ㅓ ㅕ ㅣ ㅐ ㅔ ㅒ ㅖ ㅙ ㅚ ㅝ ㅞ ㅟ ㅢ
  받침:X     -- a trailing consonant (ㄴ ㄹ ㅁ ㅇ ㅂ ㄱ ㅅ ..., rare ones pooled)
Words come from the existing corpus (so their glyphs are already well covered
and the ending is the only new variable) plus a few generic demo inputs that
failed. About a quarter of the batch is 2-3 word phrases without punctuation,
covering the space-syllable-EOT path single words never exercise. Ending mix:
~50% bare, ~34% ".", ~8% "!", ~8% "?" (the "?" only on 까/요/니/나/지/죠/네
endings) so one-word lines never teach "short line means no period".

No LLM calls; deterministic for a given --seed.

Usage (from handwriting-generation/):
  python generate_endings_batch.py                     # -> ../collection-app/curated/endings-2026-10.jsonl
  python generate_endings_batch.py --count 480 --seed 1 --out ../collection-app/curated/endings-2026-10.jsonl

Then import at priority so it is served before the exhausted pool:
  cd ../collection-app && PRIORITY=2 bash import_sentences.sh curated/endings-2026-10.jsonl
Give the batch its own "source" (default synthetic-line-endings) so the
leak-free val comparison at retrain time can isolate it.

Pair with the training-side fix: STRIP_PERIOD_P in train.py (period-stripping
augmentation of existing recordings), which de-confounds "bare ending" from
"short line" -- this batch alone would teach only short bare lines.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_POOL = REPO / "data" / "export" / "sentences.jsonl"
CURATED_DIR = REPO / "collection-app" / "curated"

# --- Hangul helpers (mirrors generate_sentences.py / collection-app hangul.ts) --

SYLLABLE_BASE = 0xAC00
LEADING_COUNT, VOWEL_COUNT, TRAILING_COUNT = 19, 21, 28
N_COUNT = VOWEL_COUNT * TRAILING_COUNT
SYLLABLE_COUNT = LEADING_COUNT * N_COUNT
TRAILING_NAMES = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
    "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
# Vowel index (ㅏ=0 .. ㅣ=20) -> which stroke is written last, by standard
# stroke order: ㅏ is vertical-then-tick, ㅗ is tick-then-bar, ㅓ is tick-then-
# vertical, compound vowels end on their second element.
VOWEL_TICK = {0, 2, 9, 13, 17}  # ㅏ ㅑ ㅘ ㅜ ㅠ
VOWEL_BAR = {8, 12, 18}  # ㅗ ㅛ ㅡ
# Common 받침 keep their own class; the rest (doubles, clusters, ㄷㅈㅊㅋㅌㅍㅎ) pool.
MAJOR_TRAILING = {"ㄴ", "ㄹ", "ㅁ", "ㅇ", "ㅂ", "ㄱ", "ㅅ"}

QUESTION_ENDINGS = set("까요니나지죠네")

# Generic demo inputs from the thumbs-down log that hit this bug (no personal
# names). They are the cases people actually type into hangul.ink.
DEMO_WORDS = [
    "안녕하세요", "고양이", "커피", "감사합니다", "반갑습니다", "사랑해", "좋아요",
    "훌륭해요", "축하해요", "세종대왕", "화이팅", "보고싶다", "잘자", "맞아", "뭐해",
    "강아지", "한글", "서울", "행복", "친구", "엄마", "아빠", "고마워", "미안해",
]


def is_syllable(ch: str) -> bool:
    return 0 <= ord(ch) - SYLLABLE_BASE < SYLLABLE_COUNT


def ending_class(word: str) -> str:
    """Shape class of the last stroke of the word's final syllable."""
    code = ord(word[-1]) - SYLLABLE_BASE
    trailing = code % TRAILING_COUNT
    if trailing:
        name = TRAILING_NAMES[trailing]
        return f"받침:{name}" if name in MAJOR_TRAILING else "받침:기타"
    vowel = (code % N_COUNT) // TRAILING_COUNT
    if vowel in VOWEL_TICK:
        return "tick"
    if vowel in VOWEL_BAR:
        return "bar"
    return "vertical"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --- Corpus ---------------------------------------------------------------------


def load_corpus_texts() -> list[str]:
    texts: list[str] = []
    for f in [DEFAULT_POOL, *sorted(CURATED_DIR.glob("*.jsonl"))]:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                texts.append(normalize(json.loads(line)["text"]))
    return texts


PARTICLE_FINALS = set("가를을는은의에도로와과서만")  # fallback-pool filter only


class Candidates:
    """Clause-final words and phrases from the corpus, with the punctuation each
    was observed before. Taking the *last* words of a clause keeps the batch
    natural -- these are the forms Korean lines actually end on (좋다, 했어요,
    어때요, a noun before a period) -- rather than mid-sentence fragments."""

    def __init__(self) -> None:
        self.word_freq: Counter = Counter()
        self.word_marks: dict[str, set[str]] = defaultdict(set)
        self.phrase_freq: Counter = Counter()
        self.phrase_marks: dict[str, set[str]] = defaultdict(set)
        self.any_word_freq: Counter = Counter()

    def add(self, text: str, max_phrase_len: int) -> None:
        for clause, mark in re.findall(r"([^.,!?]+)([.,!?]*)", text):
            words = [w for w in clause.split(" ") if w]
            if not words or any(not all(is_syllable(c) for c in w) for w in words):
                continue
            self.any_word_freq.update(words)
            m = mark[:1] if mark else ""
            if m not in ".!?":
                # Clause before a comma (or an unterminated tail): its last word
                # is a connective form (-며, -지만, -는데) that no line ends on.
                continue
            w = words[-1]
            self.word_freq[w] += 1
            self.word_marks[w].add(m)
            for n in (2, 3):
                if len(words) >= n:
                    p = " ".join(words[-n:])
                    if len(p) <= max_phrase_len:
                        self.phrase_freq[p] += 1
                        self.phrase_marks[p].add(m)


# --- Selection --------------------------------------------------------------------


def class_quota(n: int) -> dict[str, int]:
    """How many single words per ending class. Ticks get the most: that is the
    stroke the bug eats most visibly (다→디, 아→이, 야→아)."""
    weights = {
        "tick": 0.28,
        "bar": 0.20,
        "vertical": 0.16,
        "받침:ㄴ": 0.06,
        "받침:ㄹ": 0.05,
        "받침:ㅁ": 0.05,
        "받침:ㅇ": 0.06,
        "받침:ㅂ": 0.04,
        "받침:ㄱ": 0.04,
        "받침:ㅅ": 0.03,
        "받침:기타": 0.03,
    }
    quota = {k: int(round(v * n)) for k, v in weights.items()}
    diff = n - sum(quota.values())
    quota["tick"] += diff
    return quota


def pick_punct(line: str, marks: set[str], rng: random.Random, mix: dict[str, float]) -> str:
    r = rng.random()
    acc = 0.0
    choice = ""
    for mark, p in mix.items():
        acc += p
        if r < acc:
            choice = mark
            break
    if choice == "?" and "?" not in marks and line[-1] not in QUESTION_ENDINGS:
        choice = "."
    return line + choice


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=480, help="lines to produce")
    ap.add_argument("--phrase-frac", type=float, default=0.25, help="share of 2-3 word phrases")
    ap.add_argument("--out", type=Path, default=CURATED_DIR / f"endings-{time.strftime('%Y-%m')}.jsonl")
    ap.add_argument("--source", default="synthetic-line-endings")
    ap.add_argument("--min-syllables", type=int, default=2, help="shortest single word (DEMO_WORDS exempt)")
    ap.add_argument("--max-syllables", type=int, default=6)
    ap.add_argument("--min-freq", type=int, default=2, help="a word must occur this often in the corpus")
    ap.add_argument("--max-len", type=int, default=16, help="phrase length cap in characters")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true", help="print the selection, write nothing")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    mix = {"": 0.50, ".": 0.34, "!": 0.08, "?": 0.08}

    texts = load_corpus_texts()
    if not texts:
        sys.exit(f"no corpus at {DEFAULT_POOL}; run download_export.sh first")
    existing = set(texts)
    cands = Candidates()
    for t in texts:
        cands.add(t, args.max_len)

    def weighted_pop(pool: list[tuple[str, float]]) -> str:
        total = sum(wt for _, wt in pool)
        r = rng.random() * total
        acc = 0.0
        for j, (w, wt) in enumerate(pool):
            acc += wt
            if r <= acc:
                break
        return pool.pop(j)[0]

    # Primary pool: clause-final words, weighted by sqrt(frequency). Fallback
    # (only for classes the final-word pool cannot fill): any corpus word seen
    # >= min-freq times that does not end in a bare particle.
    by_class: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for w, n in cands.word_freq.items():
        if args.min_syllables <= len(w) <= args.max_syllables:
            by_class[ending_class(w)].append((w, n**0.5))
    fallback: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for w, n in cands.any_word_freq.items():
        if (
            n >= args.min_freq
            and args.min_syllables <= len(w) <= args.max_syllables
            and w not in cands.word_freq
            and not (len(w) >= 2 and w[-1] in PARTICLE_FINALS)
        ):
            fallback[ending_class(w)].append((w, n**0.5))

    n_phrases = int(round(args.count * args.phrase_frac))
    n_words = args.count - n_phrases
    quota = class_quota(n_words)

    chosen_words: list[str] = []
    used: set[str] = set()
    for w in DEMO_WORDS:  # demo failures first, counted against their class
        if w not in used and ending_class(w) in quota:
            chosen_words.append(w)
            used.add(w)
            quota[ending_class(w)] -= 1
    short: Counter = Counter()
    from_fallback = 0
    for cls, k in quota.items():
        pool = [(w, wt) for w, wt in by_class.get(cls, []) if w not in used]
        fb = [(w, wt) for w, wt in fallback.get(cls, []) if w not in used]
        for _ in range(max(0, k)):
            if pool:
                w = weighted_pop(pool)
            elif fb:
                w = weighted_pop(fb)
                from_fallback += 1
            else:
                short[cls] += 1
                continue
            chosen_words.append(w)
            used.add(w)

    # Phrases: the last 2-3 words of a clause, round-robin over ending classes
    # so ticks/bars are not starved by 받침-heavy nouns.
    by_cls_phr: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for p, n in cands.phrase_freq.items():
        if p not in used:
            by_cls_phr[ending_class(p)].append((p, n**0.5))
    chosen_phrases: list[str] = []
    order = ["tick", "bar", "vertical", "받침:ㄴ", "받침:ㄹ", "받침:ㅁ", "받침:ㅇ", "받침:ㅂ", "받침:ㄱ", "받침:ㅅ", "받침:기타"]
    while len(chosen_phrases) < n_phrases and any(by_cls_phr.values()):
        for cls in order:
            if by_cls_phr[cls] and len(chosen_phrases) < n_phrases:
                p = weighted_pop(by_cls_phr[cls])
                chosen_phrases.append(p)
                used.add(p)

    lines = [pick_punct(w, cands.word_marks.get(w, set()), rng, mix) for w in chosen_words]
    lines += [pick_punct(p, cands.phrase_marks.get(p, set()), rng, mix) for p in chosen_phrases]
    # Never duplicate a text already in the pool (incl. punctuation), and keep
    # the batch itself free of repeats.
    seen: set[str] = set()
    final: list[str] = []
    for ln in lines:
        if ln in existing or ln in seen:
            continue
        seen.add(ln)
        final.append(ln)
    rng.shuffle(final)

    # Report.
    cls_hist = Counter(ending_class(ln.rstrip(".!?")) for ln in final)
    punct_hist = Counter((ln[-1] if ln[-1] in ".!?" else "bare") for ln in final)
    n_phr = sum(1 for ln in final if " " in ln)
    print(f"{len(final)} lines ({len(final) - n_phr} words, {n_phr} phrases); "
          f"mean length {sum(map(len, final)) / max(1, len(final)):.1f} chars")
    print("ending class:", " ".join(f"{k}:{v}" for k, v in sorted(cls_hist.items(), key=lambda kv: -kv[1])))
    print("punctuation :", " ".join(f"{k}:{v}" for k, v in punct_hist.most_common()))
    bare_by_cls = Counter(ending_class(ln) for ln in final if ln[-1] not in ".!?")
    print("bare by class:", " ".join(f"{k}:{v}" for k, v in sorted(bare_by_cls.items(), key=lambda kv: -kv[1])))
    if from_fallback:
        print(f"words from the non-final fallback pool: {from_fallback}")
    if short:
        print("classes short of quota:", dict(short), file=sys.stderr)
    print("sample:", " | ".join(final[:24]))

    if args.dry_run:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for ln in final:
            out.write(json.dumps({"text": ln, "source": args.source}, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(final)} lines -> {args.out}")
    print(f"import with:\n  cd {CURATED_DIR.parent} && PRIORITY=2 bash import_sentences.sh curated/{args.out.name}")


if __name__ == "__main__":
    main()
