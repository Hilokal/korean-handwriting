"""Fetch clean Korean reference sentences from Wikipedia for handwriting data collection.

Pulls plain-text article extracts from the Korean Wikipedia API, splits them
into sentences, and keeps only sentences suitable for copying by hand:
pure Hangul syllables plus spaces and basic punctuation, within a length range.

Usage:
    python fetch_reference_texts.py                  # fetch the default page list
    python fetch_reference_texts.py 한글 세종대왕      # fetch specific pages
    python fetch_reference_texts.py --featured --good  # all featured + good articles
    python fetch_reference_texts.py --category "분류:한국의 산" --min-len 5

Output:
    ../data/reference-texts/<page>.txt   one sentence per line, per source page
    ../data/reference-texts/all.jsonl    every sentence with its source page

Also prints Hangul syllable / jamo coverage stats, since the model is
conditioned on jamo and coverage of the syllable space is what matters.
"""

import argparse
import http.cookiejar
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from tokenizer import tokenize

API_URL = "https://ko.wikipedia.org/w/api.php"
USER_AGENT = "handwriting-reference-texts/0.1 (personal research project)"

DEFAULT_PAGES = [
    # History / language
    "한글", "세종", "훈민정음", "한국어", "이순신", "조선", "고구려", "신라",
    # Places
    "대한민국", "서울특별시", "제주특별자치도", "백두산", "한라산", "부산광역시",
    # Culture / food
    "김치", "비빔밥", "불고기", "한복", "판소리", "아리랑", "태권도", "탈춤",
    # Nature / science
    "호랑이", "은행나무", "물리학", "컴퓨터", "세포", "지구", "태양",
    # Literature
    "김소월", "윤동주",
]

# Curated categories of well-written articles, for bulk fetching.
FEATURED_CATEGORY = "분류:알찬 글"
GOOD_CATEGORY = "분류:좋은 글"

# Sentences must consist only of these characters after normalization:
# Hangul syllables, space, and punctuation a person can naturally write.
ALLOWED_PUNCT = ".,!?"
HANGUL_SYLLABLE = re.compile(r"[가-힣]")
ALLOWED_SENTENCE = re.compile(rf"^[가-힣 {re.escape(ALLOWED_PUNCT)}]+$")

# Plaintext extracts still contain section headings like "== 역사 ==".
SECTION_HEADING = re.compile(r"^=+ .* =+$", re.MULTILINE)
# Parentheticals usually hold hanja/romanizations ("한글(韓㐎)") — drop them
# rather than rejecting the whole sentence. Innermost-first, applied repeatedly.
PARENTHETICAL = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
# Title brackets (《훈민정음》) wrap a word that is part of the sentence — keep
# the word, drop the brackets. Non-Hangul titles are filtered out later anyway.
TITLE_BRACKETS = re.compile(r"[〈〉《》「」『』]")
# Split on sentence-ending punctuation followed by whitespace.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# Cookie-aware opener so a bot-password login session persists across requests.
_OPENER = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
)


def api_get(params: dict, max_retries: int = 8, post: bool = False) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    url = API_URL if post else f"{API_URL}?{query}"
    data = query.encode() if post else None
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    for attempt in range(max_retries):
        try:
            with _OPENER.open(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == max_retries - 1:
                raise
            delay = float(e.headers.get("Retry-After") or 2**attempt)
            print(f"   rate limited, retrying in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError, OSError) as e:
            if attempt == max_retries - 1:
                raise
            # Patient backoff (up to ~5 min between tries) so a laptop
            # sleep / network switch doesn't kill an overnight run.
            delay = min(300, 5 * 2**attempt)
            print(f"   network error ({e}), retrying in {delay}s", file=sys.stderr)
            time.sleep(delay)


def login(username: str, password: str) -> None:
    """Log in with a bot password (Special:BotPasswords) for higher rate limits."""
    token = api_get({"action": "query", "meta": "tokens", "type": "login"})[
        "query"
    ]["tokens"]["logintoken"]
    result = api_get(
        {
            "action": "login",
            "lgname": username,
            "lgpassword": password,
            "lgtoken": token,
        },
        post=True,
    )["login"]
    if result["result"] != "Success":
        sys.exit(f"login failed: {result}")
    print(f"logged in as {result['lgusername']}")


def fetch_category_members(category: str) -> list[str]:
    """Return all article titles in a category (follows continuation)."""
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmnamespace": 0,  # articles only, not subcategories/templates
        "cmlimit": 500,
    }
    while True:
        data = api_get(params)
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            return titles
        params.update(data["continue"])


def fetch_extract(title: str) -> str:
    """Return the plain-text extract of a Wikipedia page (empty if missing)."""
    data = api_get(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
        }
    )
    pages = data["query"]["pages"]
    if not pages or "missing" in pages[0]:
        return ""
    return pages[0].get("extract", "")


def extract_sentences(text: str, min_len: int, max_len: int) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    text = SECTION_HEADING.sub("", text)
    prev = None
    while prev != text:
        prev = text
        text = PARENTHETICAL.sub("", text)
    text = TITLE_BRACKETS.sub("", text)

    sentences = []
    for paragraph in text.split("\n"):
        for sentence in SENTENCE_END.split(paragraph.strip()):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not (min_len <= len(sentence) <= max_len):
                continue
            if not ALLOWED_SENTENCE.match(sentence):
                continue
            # Require it to be mostly Hangul, not mostly punctuation.
            if len(HANGUL_SYLLABLE.findall(sentence)) < min_len // 2:
                continue
            sentences.append(sentence)
    return sentences


def coverage_report(sentences: list[str]) -> str:
    syllables = Counter(
        ch for s in sentences for ch in s if HANGUL_SYLLABLE.match(ch)
    )
    leading, vowel, trailing = set(), set(), set()
    no_trailing = 0
    for ch in syllables:
        l, v, t = (int(i) for i in tokenize(ch))
        leading.add(l)
        vowel.add(v)
        trailing.add(t)
        if t == 0:
            no_trailing += 1
    lines = [
        f"sentences:          {len(sentences)}",
        f"total syllables:    {sum(syllables.values())}",
        f"unique syllables:   {len(syllables)} (of 11172 possible)",
        f"leading jamo seen:  {len(leading)}/19",
        f"vowel jamo seen:    {len(vowel)}/21",
        f"trailing jamo seen: {len(trailing)}/28 ({no_trailing} unique syllables with no 받침)",
        f"most common:        {' '.join(ch for ch, _ in syllables.most_common(20))}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", nargs="*", default=None, help="Wikipedia page titles")
    parser.add_argument("--min-len", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=60)
    parser.add_argument(
        "--featured", action="store_true", help=f"include all pages in {FEATURED_CATEGORY}"
    )
    parser.add_argument(
        "--good", action="store_true", help=f"include all pages in {GOOD_CATEGORY}"
    )
    parser.add_argument(
        "--category", action="append", default=[], help="include all pages in a category"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip pages whose .txt already exists and append to all.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "reference-texts",
    )
    args = parser.parse_args()

    username = os.environ.get("WIKI_USERNAME")
    password = os.environ.get("WIKI_BOT_PASSWORD")
    if username and password:
        login(username, password)
    else:
        print("no WIKI_USERNAME/WIKI_BOT_PASSWORD set; fetching anonymously")

    categories = list(args.category)
    if args.featured:
        categories.append(FEATURED_CATEGORY)
    if args.good:
        categories.append(GOOD_CATEGORY)

    pages = list(args.pages) if args.pages else []
    if not pages and not categories:
        pages = list(DEFAULT_PAGES)
    for category in categories:
        members = fetch_category_members(category)
        print(f"{category}: {len(members)} pages")
        pages += [p for p in members if p not in pages]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_sentences: list[str] = []
    seen: set[str] = set()

    jsonl_path = args.out_dir / "all.jsonl"
    if args.resume and jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                seen.add(text)
                all_sentences.append(text)
        print(f"resuming with {len(all_sentences)} existing sentences")

    with open(jsonl_path, "a" if args.resume else "w", encoding="utf-8") as jsonl:
        for i, title in enumerate(pages, 1):
            page_file = args.out_dir / f"{title.replace('/', '_')}.txt"
            if args.resume and page_file.exists():
                continue
            extract = fetch_extract(title)
            if not extract:
                print(f"!! page not found: {title}", file=sys.stderr)
                continue
            sentences = extract_sentences(extract, args.min_len, args.max_len)
            # Dedupe across pages but keep per-page files complete.
            fresh = [s for s in sentences if s not in seen]
            seen.update(fresh)
            all_sentences.extend(fresh)

            for s in fresh:
                jsonl.write(
                    json.dumps({"text": s, "source": title}, ensure_ascii=False) + "\n"
                )
            jsonl.flush()
            # Written last: this is the marker --resume uses to skip the page.
            # If we die before this, the page is re-fetched and its sentences
            # deduped against all.jsonl, so nothing is lost or duplicated.
            page_file.write_text("\n".join(sentences) + "\n", encoding="utf-8")
            print(
                f"[{i}/{len(pages)}] {title}: {len(sentences)} sentences "
                f"({len(fresh)} new, {len(all_sentences)} total)"
            )
            time.sleep(0.5)  # be polite to the API

    print()
    print(coverage_report(all_sentences))
    print(f"\nwrote {args.out_dir}/all.jsonl and per-page .txt files")


if __name__ == "__main__":
    main()
