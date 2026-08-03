"""Text utilities for exploratory analysis in ComplaintIQ."""

from __future__ import annotations

import re
from collections import Counter
from itertools import pairwise
from typing import Any

# A tiny stoplist keeps the token views readable without pulling in a modeling library.
# This is descriptive counting (EDA), not feature extraction.
STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "at",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "them",
    "my",
    "our",
    "your",
    "their",
    "me",
    "us",
    "as",
    "by",
    "with",
    "from",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "not",
    "no",
    "so",
    "if",
    "but",
    "then",
    "than",
    "there",
    "here",
    "when",
    "what",
    "which",
    "who",
    "will",
    "would",
    "can",
    "could",
    "should",
    "about",
    "into",
    "out",
    "up",
    "down",
    "over",
    "under",
    "again",
    "mine",
}


def tokenize(text: Any) -> list[str]:
    """
    Lowercase word tokens of >=3 letters, minus stopwords.

    Args:
        text: String to tokenize (coerced to str if needed).

    Returns:
        list: List of token strings.
    """
    return [
        w for w in re.findall(r"[a-z']{3,}", str(text).lower()) if w not in STOPWORDS
    ]


def top_document_terms(texts: Any, k: int = 15) -> list[tuple[str, int]]:
    """
    Most common terms by DOCUMENT frequency (count each term once per document).

    Avoids inflating counts of terms that appear many times in a single verbose document.

    Args:
        texts: Iterable of text strings.
        k: Number of top terms to return (default 15).

    Returns:
        list: List of (term, count) tuples, sorted by count descending.
    """
    term_counts: Counter[str] = Counter()
    for document in texts:
        term_counts.update(set(tokenize(document)))
    return term_counts.most_common(k)


def top_bigrams(texts: Any, k: int = 15) -> list[tuple[str, int]]:
    """
    Most common adjacent word pairs (corpus frequency).

    Args:
        texts: Iterable of text strings.
        k: Number of top bigrams to return (default 15).

    Returns:
        list: List of (bigram_string, count) tuples, sorted by count descending.
    """
    bigram_counts: Counter[tuple[str, str]] = Counter()
    for document in texts:
        tokens = tokenize(document)
        bigram_counts.update(pairwise(tokens))
    return [(" ".join(bigram), count) for bigram, count in bigram_counts.most_common(k)]
