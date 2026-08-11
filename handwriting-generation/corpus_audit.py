#!/usr/bin/env python3
"""Audit sentence pools for coverage holes along the axes the model conditions on.

A hole is not "a rare syllable" -- it is a rare cell in the model's actual
conditioning space. The 요 lesson (2026-08-11): 요 was fine mid-word but had
~2 examples in line-final position, so the model garbled the most common
conversational ending in Korean. This script measures the axes where that
class of gap can hide:

  1. Jamo marginals -- leading/vowel/trailing embeddings are shared across
     syllables, so a jamo's total gradient is its corpus share. Zero or
     near-zero rows are untrained embeddings.
  2. Line-final / line-initial syllables -- position-conditioned dynamics
     (the EOT transition is trained per final syllable; line starts set the
     initial stroke context).
  3. Symbol tokens (space . , ! ?) -- each is a conditioning unit with its
     own embedding.
  4. Optional reference divergence: given a file of deployment-register text
     (e.g. demo-feedback inputs -- what users actually ask the model to
     write), report what is common there but starved in training. This is
     the heuristic that would have found the 요 hole automatically.

Usage:
  python corpus_audit.py FILE [FILE ...] [--reference FILE] [--top N]

FILEs are .jsonl ({"text": ...} per line) or plain text (one line per
sentence). Multiple files are pooled (e.g. the Wikipedia pool + a curated
batch).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

LEADING = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
VOWELS = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
TRAILING = ["∅"] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
SYMBOLS = [" ", ".", ",", "!", "?"]

SYLLABLE_BASE = 0xAC00
SYLLABLE_COUNT = 19 * 21 * 28


def decompose(ch: str):
    idx = ord(ch) - SYLLABLE_BASE
    if 0 <= idx < SYLLABLE_COUNT:
        return idx // (21 * 28), (idx % (21 * 28)) // 28, idx % 28
    return None


def load_texts(paths: list[str]) -> list[str]:
    texts = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                texts.append(json.loads(line)["text"].strip())
            else:
                texts.append(line)
    return texts


class Stats:
    def __init__(self, texts: list[str]):
        self.n_sentences = len(texts)
        self.jamo = [Counter(), Counter(), Counter()]  # leading, vowel, trailing
        self.syllables = Counter()
        self.finals = Counter()  # line-final syllable
        self.final_vowels = Counter()
        self.initials = Counter()
        self.symbols = Counter()
        for t in texts:
            hangul = [ch for ch in t if decompose(ch) is not None]
            for ch in t:
                d = decompose(ch)
                if d is None:
                    if ch in SYMBOLS:
                        self.symbols[ch] += 1
                    continue
                self.syllables[ch] += 1
                for slot, idx in enumerate(d):
                    self.jamo[slot][idx] += 1
            if hangul:
                self.finals[hangul[-1]] += 1
                self.final_vowels[VOWELS[decompose(hangul[-1])[1]]] += 1
                self.initials[hangul[0]] += 1
        self.total_syllables = sum(self.syllables.values())


def share(counter: Counter, key, total: int) -> str:
    n = counter.get(key, 0)
    return f"{n} ({100 * n / total:.2f}%)" if total else "0"


def report(stats: Stats, top: int) -> None:
    print(f"pool: {stats.n_sentences} sentences, {stats.total_syllables} syllables\n")

    names = [("leading", LEADING), ("vowel", VOWELS), ("trailing", TRAILING)]
    for slot, (label, alphabet) in enumerate(names):
        counts = [(alphabet[i], stats.jamo[slot].get(i, 0)) for i in range(len(alphabet))]
        counts.sort(key=lambda kv: kv[1])
        starved = [
            f"{ch} {n} ({100 * n / stats.total_syllables:.2f}%)" for ch, n in counts[:top]
        ]
        print(f"most-starved {label} jamo:  " + "  ".join(starved))

    print(f"\nline-final syllables ({stats.n_sentences} lines):")
    for ch, n in stats.finals.most_common(top):
        print(f"  {ch}: {n} ({100 * n / stats.n_sentences:.1f}%)")
    seen_final_vowels = set(stats.final_vowels)
    never_final = [v for v in VOWELS if v not in seen_final_vowels]
    print(f"  vowels NEVER line-final: {' '.join(never_final) or '(none)'}")

    print("\nsymbol tokens:")
    for s in SYMBOLS:
        label = "space" if s == " " else s
        print(f"  {label}: {stats.symbols.get(s, 0)}")


def divergence(train: Stats, ref: Stats, top: int) -> None:
    print("\n=== reference-register divergence (common there, starved here) ===")

    def ratio_table(name, ref_counter, train_counter, ref_total, train_total, min_ref_share):
        rows = []
        for key, n in ref_counter.items():
            ref_share = n / ref_total
            if ref_share < min_ref_share:
                continue  # too rare in the reference to matter
            train_share = train_counter.get(key, 0) / train_total
            rows.append((ref_share / max(train_share, 1e-9), key, ref_share, train_share))
        rows.sort(reverse=True)
        if not rows:
            return
        print(f"\n{name} (ref share -> train share, worst ratios first):")
        for r, key, rs, ts in rows[:top]:
            flag = "  <-- HOLE" if r > 5 else ""
            print(f"  {key}: {100 * rs:.2f}% -> {100 * ts:.2f}%  ({r:.0f}x){flag}")

    ratio_table(
        "line-final syllables",
        ref.finals, train.finals, max(ref.n_sentences, 1), max(train.n_sentences, 1),
        min_ref_share=0.01,
    )
    ratio_table(
        "syllables anywhere",
        ref.syllables, train.syllables, max(ref.total_syllables, 1), max(train.total_syllables, 1),
        min_ref_share=0.003,
    )
    for slot, label in ((0, "leading jamo"), (1, "vowel jamo"), (2, "trailing jamo")):
        alphabet = [LEADING, VOWELS, TRAILING][slot]
        ratio_table(
            label,
            Counter({alphabet[k]: v for k, v in ref.jamo[slot].items()}),
            Counter({alphabet[k]: v for k, v in train.jamo[slot].items()}),
            max(ref.total_syllables, 1), max(train.total_syllables, 1),
            min_ref_share=0.005,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="training-pool sentence files (.jsonl or .txt)")
    ap.add_argument("--reference", help="deployment-register text to diff against")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    train = Stats(load_texts(args.files))
    report(train, args.top)
    if args.reference:
        divergence(train, Stats(load_texts([args.reference])), args.top)


if __name__ == "__main__":
    main()
