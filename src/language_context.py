"""
language_context.py - lightweight next-vocabulary language context.

This module is intentionally small and dependency-free. It borrows the ASR
decoding idea of combining an acoustic/visual model with an external language
model, but keeps the implementation suitable for Jetson-class edge devices:
smoothed word n-grams, optional phrase corpus files, and deterministic scoring.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
import re
from typing import Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover - runtime fallback when pandas is absent.
    pd = None

import feature_engine as fe

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, "dataset_parquets")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

DEFAULT_CORPUS_PATHS = [
    os.path.join(MODEL_DIR, "bisindo_phrases.txt"),
    os.path.join(MODEL_DIR, "vocab_corpus.txt"),
    os.path.join(ROOT_DIR, "corpus", "bisindo_phrases.txt"),
]
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "language_context.json")

TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def normalize_token(token: str) -> str:
    return str(token).strip().lower().replace(" ", "_")


def tokenize_phrase(text: str, vocabulary: set[str] | None = None) -> list[str]:
    tokens = [normalize_token(match.group(0)) for match in TOKEN_RE.finditer(str(text).lower())]
    if vocabulary:
        tokens = [tok for tok in tokens if tok in vocabulary]
    return tokens


def _read_phrase_corpus(paths: Iterable[str], vocabulary: set[str]) -> list[list[str]]:
    phrases: list[list[str]] = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = tokenize_phrase(line, vocabulary)
                if tokens:
                    phrases.append(tokens)
    return phrases


def _vocabulary_from_parquets() -> tuple[list[str], Counter]:
    vocab = []
    counts: Counter = Counter()
    if not os.path.exists(DATABASE_DIR) or pd is None:
        return vocab, counts
    for file_name in sorted(os.listdir(DATABASE_DIR)):
        if not file_name.endswith(".parquet"):
            continue
        token = normalize_token(file_name.replace(".parquet", ""))
        if token == "idle":
            continue
        vocab.append(token)
        path = os.path.join(DATABASE_DIR, file_name)
        try:
            df = pd.read_parquet(path)
            df = fe.filter_current_feature_rows(df)
            if not df.empty and "video_id" in df.columns:
                counts[token] += int(df["video_id"].nunique())
        except Exception:
            counts[token] += 1
    return sorted(set(vocab)), counts


@dataclass
class Suggestion:
    token: str
    score: float
    lm_logprob: float
    visual_logprob: float | None = None


class NGramVocabularyLanguageModel:
    def __init__(self, order: int = 3, alpha: float = 0.15):
        self.order = int(max(1, order))
        self.alpha = float(max(alpha, 1e-6))
        self.vocabulary: list[str] = []
        self.unigram_counts: Counter = Counter()
        self.context_counts: dict[int, Counter] = defaultdict(Counter)
        self.next_counts: dict[int, Counter] = defaultdict(Counter)
        self.trained_from_corpus = False

    def fit(self, phrases: Iterable[Iterable[str]], vocabulary: Iterable[str], fallback_counts: Counter | None = None):
        self.vocabulary = sorted({normalize_token(v) for v in vocabulary if normalize_token(v) and normalize_token(v) != "idle"})
        self.unigram_counts = Counter()
        self.context_counts = defaultdict(Counter)
        self.next_counts = defaultdict(Counter)
        fallback_counts = fallback_counts or Counter()

        phrase_count = 0
        for phrase in phrases:
            tokens = [normalize_token(tok) for tok in phrase if normalize_token(tok) in self.vocabulary]
            if not tokens:
                continue
            phrase_count += 1
            padded = ["<s>"] * (self.order - 1) + tokens + ["</s>"]
            for idx in range(self.order - 1, len(padded)):
                token = padded[idx]
                if token != "</s>":
                    self.unigram_counts[token] += 1
                for n in range(1, self.order + 1):
                    context = tuple(padded[max(0, idx - n + 1):idx])
                    if len(context) != n - 1:
                        continue
                    self.context_counts[n][context] += 1
                    self.next_counts[n][context + (token,)] += 1

        self.trained_from_corpus = phrase_count > 0
        if not self.trained_from_corpus:
            for token in self.vocabulary:
                self.unigram_counts[token] = max(int(fallback_counts.get(token, 0)), 1)
        return self

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "alpha": self.alpha,
            "vocabulary": self.vocabulary,
            "unigram_counts": dict(self.unigram_counts),
            "trained_from_corpus": self.trained_from_corpus,
            "contexts": {
                str(n): {
                    "\t".join(context): int(count)
                    for context, count in counter.items()
                }
                for n, counter in self.context_counts.items()
            },
            "next_counts": {
                str(n): {
                    "\t".join(key): int(count)
                    for key, count in counter.items()
                }
                for n, counter in self.next_counts.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "NGramVocabularyLanguageModel":
        model = cls(order=int(payload.get("order", 3)), alpha=float(payload.get("alpha", 0.15)))
        model.vocabulary = [normalize_token(tok) for tok in payload.get("vocabulary", [])]
        model.unigram_counts = Counter({normalize_token(k): int(v) for k, v in payload.get("unigram_counts", {}).items()})
        model.trained_from_corpus = bool(payload.get("trained_from_corpus", False))
        for n_str, counter in payload.get("contexts", {}).items():
            n = int(n_str)
            model.context_counts[n] = Counter({tuple(k.split("\t")) if k else tuple(): int(v) for k, v in counter.items()})
        for n_str, counter in payload.get("next_counts", {}).items():
            n = int(n_str)
            model.next_counts[n] = Counter({tuple(k.split("\t")) if k else tuple(): int(v) for k, v in counter.items()})
        return model

    def save(self, path: str = DEFAULT_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "NGramVocabularyLanguageModel | None":
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except Exception:
            return None

    def lm_logprob(self, prefix: Iterable[str], token: str) -> float:
        token = normalize_token(token)
        if token not in self.vocabulary:
            return -1e9
        prefix_tokens = [normalize_token(tok) for tok in prefix if normalize_token(tok)]
        vocab_size = max(len(self.vocabulary), 1)

        if self.trained_from_corpus:
            padded = ["<s>"] * (self.order - 1) + prefix_tokens
            for n in range(self.order, 0, -1):
                context = tuple(padded[-(n - 1):]) if n > 1 else tuple()
                context_total = int(self.context_counts[n].get(context, 0))
                next_count = int(self.next_counts[n].get(context + (token,), 0))
                if context_total > 0 or n == 1:
                    prob = (next_count + self.alpha) / (context_total + self.alpha * (vocab_size + 1))
                    return float(math.log(max(prob, 1e-12)))

        total = sum(self.unigram_counts.values())
        count = int(self.unigram_counts.get(token, 0))
        prob = (count + self.alpha) / (total + self.alpha * vocab_size)
        return float(math.log(max(prob, 1e-12)))

    def suggest_next(
        self,
        prefix: Iterable[str],
        visual_candidates: Iterable[tuple[str, float]] | None = None,
        top_k: int = 5,
        lm_weight: float = 0.65,
        visual_weight: float = 0.35,
    ) -> list[Suggestion]:
        visual_map = None
        if visual_candidates is not None:
            visual_map = {
                normalize_token(token): float(max(prob, 1e-8))
                for token, prob in visual_candidates
                if normalize_token(token) in self.vocabulary
            }
        candidates = list(visual_map.keys()) if visual_map else list(self.vocabulary)
        scored: list[Suggestion] = []
        for token in candidates:
            lm_lp = self.lm_logprob(prefix, token)
            visual_lp = math.log(visual_map[token]) if visual_map else None
            score = lm_weight * lm_lp
            if visual_lp is not None:
                score += visual_weight * visual_lp
            scored.append(Suggestion(token=token, score=float(score), lm_logprob=lm_lp, visual_logprob=visual_lp))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: int(max(1, top_k))]


def build_default_language_model(save: bool = True) -> NGramVocabularyLanguageModel:
    vocabulary, fallback_counts = _vocabulary_from_parquets()
    phrases = _read_phrase_corpus(DEFAULT_CORPUS_PATHS, set(vocabulary))
    model = NGramVocabularyLanguageModel(order=3, alpha=0.15).fit(phrases, vocabulary, fallback_counts)
    if save:
        model.save(DEFAULT_MODEL_PATH)
    return model


def load_default_language_model() -> NGramVocabularyLanguageModel:
    model = NGramVocabularyLanguageModel.load(DEFAULT_MODEL_PATH)
    vocabulary, fallback_counts = _vocabulary_from_parquets()
    if model is None or sorted(model.vocabulary) != sorted(vocabulary):
        return build_default_language_model(save=True)
    return model
