"""Bernoulli Naive Bayes classifier for spam probability prediction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import database


FEATURE_WORDS = [
    "限时",
    "领取",
    "点击",
    "链接",
    "免费",
    "现金",
    "礼品",
    "奖励",
    "恭喜",
    "促销",
    "优惠",
    "中奖",
]

HARD_SPAM_WORDS = ["优惠", "中奖"]

SPAM_THRESHOLD = 0.55
SPAM_LABEL = 1
NORMAL_LABEL = 0
LABELS = (SPAM_LABEL, NORMAL_LABEL)

TrainingSample = Mapping[str, object]


def extract_features(text: str) -> dict[str, int]:
    """Convert text to 0/1 keyword features for Bernoulli Naive Bayes."""

    safe_text = text or ""
    return {word: 1 if word in safe_text else 0 for word in FEATURE_WORDS}


def load_training_samples(db_path: str | Path = database.DEFAULT_DB_PATH) -> list[TrainingSample]:
    """Load training samples from the SQLite training_data table."""

    return database.list_training_data(db_path)


def _validate_samples(samples: Sequence[TrainingSample]) -> None:
    if not samples:
        raise ValueError("training samples are empty")

    labels = {int(sample["label"]) for sample in samples}
    missing_labels = set(LABELS) - labels
    if missing_labels:
        raise ValueError("training samples must contain both normal and spam labels")


def calculate_priors(samples: Sequence[TrainingSample]) -> dict[int, float]:
    """Calculate smoothed prior probabilities for spam and normal mail."""

    _validate_samples(samples)
    total_count = len(samples)
    label_counts = {
        label: sum(1 for sample in samples if int(sample["label"]) == label)
        for label in LABELS
    }

    return {
        label: (label_counts[label] + 1) / (total_count + len(LABELS))
        for label in LABELS
    }


def calculate_conditional_probabilities(
    samples: Sequence[TrainingSample],
) -> dict[int, dict[str, float]]:
    """Calculate Laplace-smoothed P(word appears | class) probabilities."""

    _validate_samples(samples)
    probabilities: dict[int, dict[str, float]] = {}

    for label in LABELS:
        class_samples = [sample for sample in samples if int(sample["label"]) == label]
        class_count = len(class_samples)
        word_probabilities: dict[str, float] = {}

        for word in FEATURE_WORDS:
            contains_word_count = sum(
                1
                for sample in class_samples
                if extract_features(str(sample["content"]))[word] == 1
            )
            word_probabilities[word] = (contains_word_count + 1) / (class_count + 2)

        probabilities[label] = word_probabilities

    return probabilities


def _log_score(
    features: Mapping[str, int],
    prior: float,
    conditional_probabilities: Mapping[str, float],
) -> float:
    score = math.log(prior)

    for word, appears in features.items():
        word_probability = conditional_probabilities[word]
        if appears:
            score += math.log(word_probability)
        else:
            score += math.log(1 - word_probability)

    return score


def _normalize_log_scores(log_spam: float, log_normal: float) -> tuple[float, float]:
    max_log = max(log_spam, log_normal)
    spam_score = math.exp(log_spam - max_log)
    normal_score = math.exp(log_normal - max_log)
    total = spam_score + normal_score

    return spam_score / total, normal_score / total


def bayes_predict(
    text: str,
    db_path: str | Path = database.DEFAULT_DB_PATH,
) -> tuple[float, float]:
    """Return (spam_probability, normal_probability) for the given text.

    Hard keyword rules are intentionally not applied here. The mail service
    layer will later decide whether to run hard rules before this classifier.
    """

    samples = load_training_samples(db_path)
    features = extract_features(text)
    priors = calculate_priors(samples)
    conditional_probabilities = calculate_conditional_probabilities(samples)

    log_spam = _log_score(
        features,
        priors[SPAM_LABEL],
        conditional_probabilities[SPAM_LABEL],
    )
    log_normal = _log_score(
        features,
        priors[NORMAL_LABEL],
        conditional_probabilities[NORMAL_LABEL],
    )

    return _normalize_log_scores(log_spam, log_normal)


def is_spam_probability(spam_probability: float, threshold: float = SPAM_THRESHOLD) -> bool:
    """Return whether a spam probability reaches the configured threshold."""

    return spam_probability >= threshold


if __name__ == "__main__":
    if not database.DEFAULT_DB_PATH.exists():
        database.initialize_database()

    samples = load_training_samples()

    print(f"训练样本数量: {len(samples)}")
    print()

    demo_texts = [
        "明天下午三点召开项目会议，请准时参加。",
        "限时领取礼品，请点击链接查看奖励。",
        "",
    ]

    for text in demo_texts:
        spam_probability, normal_probability = bayes_predict(text)
        display_text = text or "空文本"
        print("测试文本:")
        print(display_text)
        print(f"垃圾概率: {spam_probability:.2%}")
        print(f"正常概率: {normal_probability:.2%}")
        print(f"是否达到垃圾阈值: {'是' if is_spam_probability(spam_probability) else '否'}")
        print()
