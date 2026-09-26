#!/usr/bin/env python3
"""
Generate a curated batch of synthetic Korean sentences for the collection app,
targeted at the syllables a worker has recorded least.

Why: the coverage-driven assignment queue falls back to repeats once a worker
has exhausted the pool (u2 hit that in 2026-09). Curated batches imported at
priority > 0 jump the queue; this script writes one such batch, steering
Claude toward the worker's thinnest syllables so every accepted line is a
useful style sample rather than a fifth recording of a saturated sentence.

Pipeline per round:
  1. Take the worker's current per-syllable counts (from the export manifest,
     same semantics as the app's user_syllable_counts: occurrences summed over
     recorded chunks) and pick the thinnest syllables.
  2. Split them into groups; ask Claude for N sentences per group, each using
     2+ target syllables, in a mix of registers (해요체 / 합니다체 / 반말),
     6-26 characters so every line fits one Ncode notebook line (chunker
     cap is 30).
  3. Validate (pure Hangul + " .,!?", length, dedupe against the pool and
     earlier curated batches, must hit a target), append accepted lines to the
     output JSONL, and credit their syllables so the next round rotates targets.

Usage (from handwriting-generation/, venv active, ANTHROPIC_API_KEY set):
  python generate_sentences.py --count 600 --out ../collection-app/curated/thin-2026-09.jsonl
  python generate_sentences.py --fetch --count 600 ...      # pull a fresh manifest first
  python generate_sentences.py --dry-run                    # targets + prompt, no API calls

Then import with priority so it is served before the exhausted pool:
  cd ../collection-app && PRIORITY=1 bash import_sentences.sh curated/thin-2026-09.jsonl

Cost: ~count/per-call requests to claude-opus-5, a few dollars for 600 lines.
"""

from __future__ import annotations

import argparse
import getpass
import http.cookiejar
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MANIFEST = REPO / "data" / "export" / "manifest.json"
DEFAULT_POOL = REPO / "data" / "export" / "sentences.jsonl"
CURATED_DIR = REPO / "collection-app" / "curated"
ENV_FILE = REPO / ".env"

MODEL = "claude-opus-5"
BASE_URL = os.environ.get("BASE", "https://handwriting-collection.fly.dev")

# --- Hangul helpers (mirrors collection-app/server/hangul.ts) -----------------

SYLLABLE_BASE = 0xAC00
LEADING_COUNT, VOWEL_COUNT, TRAILING_COUNT = 19, 21, 28
N_COUNT = VOWEL_COUNT * TRAILING_COUNT
SYLLABLE_COUNT = LEADING_COUNT * N_COUNT
TRAILING_NAMES = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
    "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
ALLOWED_SYMBOLS = " .,!?"  # tokenizer.SYMBOLS; anything else becomes UNKNOWN


def is_syllable(ch: str) -> bool:
    return 0 <= ord(ch) - SYLLABLE_BASE < SYLLABLE_COUNT


def trailing_of(ch: str) -> str:
    return TRAILING_NAMES[(ord(ch) - SYLLABLE_BASE) % TRAILING_COUNT]


def syllable_counts(text: str) -> Counter:
    return Counter(ch for ch in text if is_syllable(ch))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --- Inputs -------------------------------------------------------------------


def load_env() -> None:
    """Fill unset env vars from the repo-root .env (KEY=value lines)."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if re.fullmatch(r"[A-Z_]+", k) and k not in os.environ:
                os.environ[k] = v


def fetch_manifest(user_id: int | None) -> list[dict]:
    """Pull the export manifest (metadata only, no dots) via the admin API."""
    load_env()
    user = os.environ.get("ADMIN_USER", "jon@jonb.org")
    password = os.environ.get("ADMIN_PASSWORD") or getpass.getpass(f"Admin password for {user}: ")
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    body = json.dumps({"username": user, "password": password}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/login", data=body, headers={"Content-Type": "application/json"}
    )
    with opener.open(req) as r:
        if r.status != 200:
            sys.exit(f"login failed: HTTP {r.status}")
    qs = f"?userId={user_id}" if user_id is not None else ""
    with opener.open(f"{BASE_URL}/api/admin/export/manifest{qs}") as r:
        return json.load(r)["recordings"]


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["recordings"] if isinstance(data, dict) else data


def coverage_from_manifest(recs: list[dict], user_id: int | None) -> Counter:
    counts: Counter = Counter()
    for r in recs:
        if user_id is not None and r.get("userId") != user_id:
            continue
        counts.update(syllable_counts(r.get("text") or r.get("sentenceText") or ""))
    return counts


def load_pool_texts(recs: list[dict]) -> set[str]:
    """Every sentence text already in play: pool export, curated batches, manifest."""
    texts: set[str] = set()
    if DEFAULT_POOL.exists():
        for line in DEFAULT_POOL.read_text().splitlines():
            if line.strip():
                texts.add(normalize(json.loads(line)["text"]))
    for f in sorted(CURATED_DIR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                texts.add(normalize(json.loads(line)["text"]))
    for r in recs:
        for k in ("sentenceText", "text"):
            if r.get(k):
                texts.add(normalize(r[k]))
    return texts


# --- Prompting ----------------------------------------------------------------

SYSTEM_PROMPT = """\
You write short, natural Korean sentences for a handwriting-collection project.
A native Korean adult will handwrite each sentence on one line of a notebook, so
the sentences must read like something a Korean person would actually say or
write: everyday life, feelings, plans, weather, food, work, family, small talk,
short observations. Fluent, idiomatic, nothing stilted or textbook-like, and no
encyclopedic or translated-sounding phrasing.

Hard constraints for every sentence:
- 6 to 26 characters long, counting spaces.
- Only Hangul syllables and these punctuation marks: space . , ! ?
  No digits, Latin letters, hanja, quotes, parentheses, hyphens, tildes,
  ellipses, emoji or line breaks. Write numbers out in Korean words if needed.
- Must contain at least one of the TARGET syllables given in the request, as
  an exact syllable block inside a natural, common Korean word. Use two or
  three targets in one sentence whenever that reads naturally, but never
  force it: naturalness beats coverage. Many targets are rare blocks from
  encyclopedia text (loanword fragments, unusual conjugations); if a target
  has no everyday word you would actually use, skip it entirely rather than
  write something odd.
- Each sentence stands alone and is a complete sentence ending in . ! or ?
- Vary the speech level across the set: roughly half 해요체, a quarter
  합니다체/formal, a quarter plain 반말 or -다 narration. Include some
  questions and exclamations, not only statements.
- No two sentences with the same structure or topic; do not reuse a target in
  the same word twice across the set.
"""

USER_TEMPLATE = """\
TARGET syllables (single Hangul blocks). Cover as many as you can naturally,
one to three per sentence; skip any that have no natural everyday word:
{targets}

Write {n} sentences. Return them in the required JSON shape, listing for each
sentence which TARGET syllables it contains.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "targets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "targets"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sentences"],
    "additionalProperties": False,
}


def build_user_prompt(targets: list[str], n: int) -> str:
    return USER_TEMPLATE.format(targets=" ".join(targets), n=n)


def call_model(client, targets: list[str], n: int, max_tokens: int) -> list[dict]:
    import anthropic

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(targets, n)}],
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        )
    except anthropic.RateLimitError as e:  # SDK already retried
        print(f"  rate limited, skipping group: {e}", file=sys.stderr)
        return []
    except anthropic.APIStatusError as e:
        print(f"  API error {e.status_code}, skipping group: {e.message}", file=sys.stderr)
        return []
    except anthropic.APIConnectionError as e:
        print(f"  connection error, skipping group: {e}", file=sys.stderr)
        return []

    if response.stop_reason != "end_turn":
        print(f"  stop_reason={response.stop_reason}, skipping group", file=sys.stderr)
        return []
    text = "".join(b.text for b in response.content if b.type == "text")
    try:
        return json.loads(text)["sentences"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  unparseable response, skipping group: {e}", file=sys.stderr)
        return []


# --- Validation ---------------------------------------------------------------


def validate(
    text: str,
    targets: set[str],
    seen: set[str],
    min_len: int,
    max_len: int,
) -> tuple[str | None, str]:
    """Return (normalized text, reason). Text is None when rejected."""
    t = normalize(text)
    if not t:
        return None, "empty"
    if any(not (is_syllable(ch) or ch in ALLOWED_SYMBOLS) for ch in t):
        return None, "non-hangul"
    if not (min_len <= len(t) <= max_len):
        return None, "length"
    if t[-1] not in ".!?":
        return None, "no-terminal-punct"
    if t in seen:
        return None, "duplicate"
    if not (targets & set(t)):
        return None, "no-target"
    return t, "ok"


# --- Main ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=600, help="accepted sentences to produce")
    ap.add_argument("--out", type=Path, default=CURATED_DIR / f"thin-{time.strftime('%Y-%m')}.jsonl")
    ap.add_argument("--source", default="synthetic-thin-syllables", help='"source" tag written per row')
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--fetch", action="store_true", help="fetch a fresh manifest from the app (writes --manifest)")
    ap.add_argument("--user-id", type=int, default=int(os.environ.get("USER_ID") or 2))
    ap.add_argument("--target-syllables", type=int, default=400, help="how many of the thinnest syllables to chase")
    ap.add_argument("--group-size", type=int, default=16, help="target syllables offered per request")
    ap.add_argument("--per-call", type=int, default=12, help="sentences requested per request")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=28, help="chunker cap is 30; keep margin")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print targets and one prompt; no API calls")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.fetch:
        recs = fetch_manifest(None)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(recs, ensure_ascii=False))
        print(f"fetched manifest: {len(recs)} recordings -> {args.manifest}")
    elif args.manifest.exists():
        recs = load_manifest(args.manifest)
    else:
        sys.exit(f"no manifest at {args.manifest}; run with --fetch or download_export.sh first")

    counts = coverage_from_manifest(recs, args.user_id)
    if not counts:
        sys.exit(f"no recordings for userId {args.user_id} in the manifest")
    pool_syllables = set()
    if DEFAULT_POOL.exists():
        for line in DEFAULT_POOL.read_text().splitlines():
            if line.strip():
                pool_syllables |= set(syllable_counts(json.loads(line)["text"]))
    for s in pool_syllables:
        counts.setdefault(s, 0)

    seen = load_pool_texts(recs)
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen.add(normalize(row["text"]))
                counts.update(syllable_counts(row["text"]))

    ranked = sorted(counts.items(), key=lambda kv: (kv[1], rng.random()))
    targets_all = [s for s, _ in ranked[: args.target_syllables]]
    hist = Counter(min(c, 5) for _, c in ranked[: args.target_syllables])
    print(
        f"worker {args.user_id}: {len(counts)} syllables tracked; chasing {len(targets_all)} thinnest "
        f"(count histogram {dict(sorted(hist.items()))}, 5=5+)"
    )
    print("thinnest 40:", " ".join(f"{s}({c})" for s, c in ranked[:40]))
    fam = Counter()
    for s, c in ranked[: args.target_syllables]:
        fam[trailing_of(s) or "∅"] += 1
    print("by 받침 family:", " ".join(f"{k}:{v}" for k, v in fam.most_common()))

    if args.dry_run:
        print("\n--- system prompt ---\n" + SYSTEM_PROMPT)
        print("--- example user prompt ---\n" + build_user_prompt(targets_all[: args.group_size], args.per_call))
        return

    import anthropic

    client = anthropic.Anthropic()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    accepted = sum(1 for line in args.out.read_text().splitlines() if line.strip()) if args.out.exists() else 0
    start_accepted = accepted
    rejects: Counter = Counter()
    hits: Counter = Counter()
    calls = 0

    with args.out.open("a", encoding="utf-8") as out:
        while accepted < args.count:
            # Re-rank every round so credited syllables rotate out of the targets.
            ranked = sorted(counts.items(), key=lambda kv: (kv[1], rng.random()))
            thin = [s for s, _ in ranked[: args.target_syllables]]
            rng.shuffle(thin)
            need = args.count - accepted
            n_groups = max(1, min(args.workers * 2, -(-need // args.per_call)))
            groups = [thin[i : i + args.group_size] for i in range(0, len(thin), args.group_size)][:n_groups]

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(lambda g: (g, call_model(client, g, args.per_call, args.max_tokens)), groups))
            calls += len(groups)

            round_ok = 0
            for group, rows in results:
                tset = set(group)
                for row in rows:
                    text, reason = validate(row.get("text", ""), tset, seen, args.min_len, args.max_len)
                    if text is None:
                        rejects[reason] += 1
                        continue
                    seen.add(text)
                    sc = syllable_counts(text)
                    counts.update(sc)
                    for s in tset & set(sc):
                        hits[s] += 1
                    out.write(json.dumps({"text": text, "source": args.source}, ensure_ascii=False) + "\n")
                    accepted += 1
                    round_ok += 1
                    if accepted >= args.count:
                        break
                if accepted >= args.count:
                    break
            out.flush()
            print(f"round: {len(groups)} calls, +{round_ok} accepted -> {accepted}/{args.count}  rejects so far {dict(rejects)}")
            if round_ok == 0:
                print("no sentences accepted this round; stopping to avoid burning credits", file=sys.stderr)
                break

    print(f"\ndone: {accepted - start_accepted} new sentences ({accepted} total in {args.out}) in {calls} API calls")
    print(f"rejected: {dict(rejects) or 'none'}")
    print(f"target syllables hit: {len(hits)} distinct; top {hits.most_common(10)}")
    still_thin = [s for s in targets_all if s not in hits]
    print(f"targets never hit: {len(still_thin)} e.g. {' '.join(still_thin[:30])}")
    print(f"\nimport with:\n  cd {CURATED_DIR.parent} && PRIORITY=1 bash import_sentences.sh curated/{args.out.name}")


if __name__ == "__main__":
    main()
