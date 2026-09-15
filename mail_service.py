"""Mail business services for classification and later mail flow handling."""

from __future__ import annotations

from datetime import datetime
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


class SendMailResult(TypedDict):
    sent_email_id: int
    received_email_id: int
    receiver_folder: str
    classification: ClassificationResult


class CorrectionResult(TypedDict):
    email_id: int
    folder: str
    is_spam: bool
    training_sample_id: int
    training_label: int
    training_text: str


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


def _insert_email_record(
    connection,
    sender_id: int,
    receiver_id: int,
    subject: str,
    content: str,
    send_time: str,
    folder: str,
    classification: ClassificationResult,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO emails (
            sender_id,
            receiver_id,
            subject,
            content,
            send_time,
            spam_probability,
            normal_probability,
            is_spam,
            folder,
            classification_method,
            classification_reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sender_id,
            receiver_id,
            subject,
            content,
            send_time,
            classification["spam_probability"],
            classification["normal_probability"],
            1 if classification["is_spam"] else 0,
            folder,
            classification["classification_method"],
            classification["classification_reason"],
        ),
    )
    return int(cursor.lastrowid)


def send_mail(
    sender_id: int,
    receiver_id: int,
    subject: str,
    content: str,
    db_path: str | Path = database.DEFAULT_DB_PATH,
) -> SendMailResult:
    """Send a mail and create both sender and receiver folder records."""

    classification = classify_mail(subject, content, db_path)
    receiver_folder = "trash" if classification["is_spam"] else "inbox"
    send_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with database.get_connection(db_path) as connection:
        sent_email_id = _insert_email_record(
            connection,
            sender_id=sender_id,
            receiver_id=receiver_id,
            subject=subject,
            content=content,
            send_time=send_time,
            folder="sent",
            classification=classification,
        )
        received_email_id = _insert_email_record(
            connection,
            sender_id=sender_id,
            receiver_id=receiver_id,
            subject=subject,
            content=content,
            send_time=send_time,
            folder=receiver_folder,
            classification=classification,
        )
        connection.commit()

    return {
        "sent_email_id": sent_email_id,
        "received_email_id": received_email_id,
        "receiver_folder": receiver_folder,
        "classification": classification,
    }


def get_inbox(
    user_id: int,
    db_path: str | Path = database.DEFAULT_DB_PATH,
):
    """Return normal received mail for the given user."""

    with database.get_connection(db_path) as connection:
        return connection.execute(
            """
            SELECT * FROM emails
            WHERE receiver_id = ? AND folder = 'inbox'
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()


def get_sent(
    user_id: int,
    db_path: str | Path = database.DEFAULT_DB_PATH,
):
    """Return sent mail records for the given user."""

    with database.get_connection(db_path) as connection:
        return connection.execute(
            """
            SELECT * FROM emails
            WHERE sender_id = ? AND folder = 'sent'
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()


def get_trash(
    user_id: int,
    db_path: str | Path = database.DEFAULT_DB_PATH,
):
    """Return spam received mail for the given user."""

    with database.get_connection(db_path) as connection:
        return connection.execute(
            """
            SELECT * FROM emails
            WHERE receiver_id = ? AND folder = 'trash'
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()


def _correct_received_mail(
    email_id: int,
    folder: str,
    label: int,
    db_path: str | Path = database.DEFAULT_DB_PATH,
) -> CorrectionResult:
    with database.get_connection(db_path) as connection:
        email = connection.execute(
            "SELECT * FROM emails WHERE id = ?",
            (email_id,),
        ).fetchone()
        if email is None:
            raise ValueError(f"email not found: {email_id}")
        if email["folder"] == "sent":
            raise ValueError("sent mail cannot be corrected")

        training_text = combine_mail_text(email["subject"], email["content"])
        database.update_email_folder_and_label_with_connection(
            connection,
            email_id=email_id,
            folder=folder,
            is_spam=label,
        )
        training_sample_id = database.add_or_update_feedback_sample_with_connection(
            connection,
            content=training_text,
            label=label,
        )
        connection.commit()

    return {
        "email_id": email_id,
        "folder": folder,
        "is_spam": bool(label),
        "training_sample_id": training_sample_id,
        "training_label": label,
        "training_text": training_text,
    }


def mark_as_spam(
    email_id: int,
    db_path: str | Path = database.DEFAULT_DB_PATH,
) -> CorrectionResult:
    """Move one received mail to trash and learn it as spam feedback."""

    return _correct_received_mail(email_id, "trash", 1, db_path)


def mark_as_normal(
    email_id: int,
    db_path: str | Path = database.DEFAULT_DB_PATH,
) -> CorrectionResult:
    """Move one received mail to inbox and learn it as normal feedback."""

    return _correct_received_mail(email_id, "inbox", 0, db_path)
