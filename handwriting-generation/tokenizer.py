import torch

LeadingCount = 19
VowelCount = 21
TrailingCount = 28

# Non-Hangul characters (space, punctuation) get their own conditioning tokens so
# their strokes attend to a real character position instead of bleeding into the
# jamo embeddings of a neighbour. Each token carries a 4th slot: 0 = "this unit is
# a Hangul syllable, use the three jamo slots"; >= 1 = one of these symbol ids.
SYMBOLS = [" ", ".", ",", "!", "?"]
UNKNOWN_SYMBOL_ID = len(SYMBOLS) + 1  # any character not listed above
SymbolCount = len(SYMBOLS) + 2  # embedding rows: id 0 (Hangul, unused) .. UNKNOWN


def _symbol_id(ch: str) -> int:
    try:
        return SYMBOLS.index(ch) + 1
    except ValueError:
        return UNKNOWN_SYMBOL_ID


def decompose_hangul_syllable(ch: str):
    SyllableBase = 0xAC00  # First syllable is "가"
    NCount = VowelCount * TrailingCount
    SyllableCount = LeadingCount * NCount

    code = ord(ch)
    syllableIndex = code - SyllableBase

    # Not a modern Hangul syllable
    if syllableIndex < 0 or syllableIndex >= SyllableCount:
        raise Exception(f"Character {ch} is not a valid Hangeul character")

    leadingIndex = syllableIndex // NCount
    vowelIndex = (syllableIndex % NCount) // TrailingCount
    trailingIndex = syllableIndex % TrailingCount

    return leadingIndex, vowelIndex, trailingIndex


def tokenize(input_string: str) -> torch.Tensor:
    """Tokenize a string to (U, 4): [leading, vowel, trailing, symbol] per unit.

    Hangul syllables decompose into their three jamo with symbol slot 0. Spaces
    and punctuation get jamo slots 0 and a symbol id >= 1 (see SYMBOLS). Anything
    else maps to the unknown-symbol id rather than raising, so unexpected
    characters in real transcripts don't crash the loader.
    """
    SyllableBase = 0xAC00
    SyllableCount = LeadingCount * VowelCount * TrailingCount

    tokens = []
    for ch in input_string:
        if 0 <= ord(ch) - SyllableBase < SyllableCount:
            leading, vowel, trailing = decompose_hangul_syllable(ch)
            tokens.append([leading, vowel, trailing, 0])
        else:
            tokens.append([0, 0, 0, _symbol_id(ch)])
    return torch.LongTensor(tokens)  # Shape: (U, 4)
