"""Mail business services for classification and later mail flow handling."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import database
from bayes import (
    FEATURE_WORDS,
    HARD_SPAM_WORDS,
    SPAM_THRESHOLD,
    bayes_predict,
)


class ClassificationResult(TypedDict):
    is_spam: bool
    spam_probability: float | None
    normal_probability: float | None
    classification_method: str
    classification_reason: str
    matched_keywords: list[str]


def combine_mail_text(subject: str | None, content: str | None) -> str:
    """Combine subject and body text safely for classification."""

    return f"{subject or ''} {content or ''}".strip()


def get_matched_feature_words(text: str | None) -> list[str]:
    """Return Bayesian feature words that appear in the text."""

    safe_text = text or ""
    return [word for word in FEATURE_WORDS if word in safe_text]


def get_matched_hard_spam_words(text: str | None) -> list[str]:
    """Return hard spam keywords that appear in the text."""

    safe_text = text or ""
    return [word for word in HARD_SPAM_WORDS if word in safe_text]


def classify_mail(
    subject: str | None,
    content: str | None,
    db_path: str | Path = database.DEFAULT_DB_PATH,
) -> ClassificationResult:
    """Classify one mail using hard spam rules first, then Bayesian probability."""

    text = combine_mail_text(subject, content)
    matched_hard_words = get_matched_hard_spam_words(text)

    if matched_hard_words:
        return {
            "is_spam": True,
            "spam_probability": None,
            "normal_probability": None,
            "classification_method": "rule",
            "classification_reason": f"命中固定垃圾关键词：{', '.join(matched_hard_words)}；贝叶斯概率未参与最终判定",
            "matched_keywords": matched_hard_words,
        }

    spam_probability, normal_probability = bayes_predict(text, db_path)
    is_spam = spam_probability >= SPAM_THRESHOLD
    threshold_text = f"{SPAM_THRESHOLD:.2%}"

    if is_spam:
        reason = f"贝叶斯垃圾概率 {spam_probability:.2%}，达到 {threshold_text} 阈值"
    else:
        reason = f"贝叶斯垃圾概率 {spam_probability:.2%}，低于 {threshold_text} 阈值"

    return {
        "is_spam": is_spam,
        "spam_probability": spam_probability,
        "normal_probability": normal_probability,
        "classification_method": "bayes",
        "classification_reason": reason,
        "matched_keywords": get_matched_feature_words(text),
    }
