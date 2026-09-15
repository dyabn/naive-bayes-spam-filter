"""SQLite persistence layer for the spam mail demo system.

This module owns database creation, seed data, and plain data access helpers.
It intentionally does not contain Bayesian classification or UI logic.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "mail.db"


class ClosingConnection(sqlite3.Connection):
    """SQLite connection that closes when leaving a with block."""

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        super().__exit__(exc_type, exc_value, traceback)
        self.close()
        return False


INITIAL_USERS = [
    ("张三", "123456"),
    ("李四", "123456"),
    ("王五", "123456"),
]


INITIAL_TRAINING_SAMPLES = [
    ("明天下午三点召开项目会议，请准时参加。", 0, "initial"),
    ("本周五提交课程设计报告，请大家提前准备。", 0, "initial"),
    ("请查看附件中的课堂资料，并完成课后练习。", 0, "initial"),
    ("今天晚上一起讨论机器学习实验进度。", 0, "initial"),
    ("老师已经发布新的作业要求，请及时查看。", 0, "initial"),
    ("部门会议改到周三上午十点举行。", 0, "initial"),
    ("请点击链接查看课程资料和实验说明。", 0, "initial"),
    ("你的图书馆借阅信息已经更新，请按时归还。", 0, "initial"),
    ("恭喜中奖，点击链接领取现金奖励。", 1, "initial"),
    ("限时领取免费礼品，请马上点击链接。", 1, "initial"),
    ("优惠活动开始，现金红包等你领取。", 1, "initial"),
    ("恭喜获得大奖，请点击链接填写银行卡信息。", 1, "initial"),
    ("免费领取礼品，限时奖励马上结束。", 1, "initial"),
    ("点击链接领取神秘现金奖励。", 1, "initial"),
    ("促销优惠火热进行，立即领取专属礼品。", 1, "initial"),
    ("中奖用户专享福利，请立即点击领取。", 1, "initial"),
]


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection and return rows as sqlite3.Row mappings."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, factory=ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_tables(connection: sqlite3.Connection) -> None:
    """Create all project tables if they do not already exist."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT
        );

        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER NOT NULL,
            receiver_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            content TEXT NOT NULL,
            send_time TEXT NOT NULL,
            spam_probability REAL NOT NULL DEFAULT 0,
            normal_probability REAL NOT NULL DEFAULT 0,
            is_spam INTEGER NOT NULL DEFAULT 0 CHECK (is_spam IN (0, 1)),
            folder TEXT NOT NULL CHECK (folder IN ('inbox', 'sent', 'trash')),
            classification_method TEXT,
            classification_reason TEXT,
            FOREIGN KEY (sender_id) REFERENCES users (id),
            FOREIGN KEY (receiver_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            label INTEGER NOT NULL CHECK (label IN (0, 1)),
            source TEXT NOT NULL DEFAULT 'initial',
            UNIQUE (content, label, source)
        );
        """
    )
    connection.commit()


def seed_users(connection: sqlite3.Connection, users: Iterable[tuple[str, str]] = INITIAL_USERS) -> None:
    """Insert demo users without duplicating existing names."""

    connection.executemany(
        "INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
        users,
    )
    connection.commit()


def seed_training_data(
    connection: sqlite3.Connection,
    samples: Iterable[tuple[str, int, str]] = INITIAL_TRAINING_SAMPLES,
) -> None:
    """Insert initial normal and spam training samples idempotently."""

    connection.executemany(
        """
        INSERT OR IGNORE INTO training_data (content, label, source)
        VALUES (?, ?, ?)
        """,
        samples,
    )
    connection.commit()


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create tables and seed the demo users and training corpus."""

    with get_connection(db_path) as connection:
        create_tables(connection)
        seed_users(connection)
        seed_training_data(connection)


def list_users(db_path: str | Path = DEFAULT_DB_PATH) -> list[sqlite3.Row]:
    """Return all users ordered by id."""

    with get_connection(db_path) as connection:
        return connection.execute("SELECT * FROM users ORDER BY id").fetchall()


def get_user_by_username(username: str, db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Row | None:
    """Find a user by username."""

    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()


def add_user(username: str, password: str = "123456", db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Create a user and return its id."""

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password),
        )
        connection.commit()
        return int(cursor.lastrowid)


def add_training_sample(
    content: str,
    label: int,
    source: str = "user_feedback",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Add a training sample and return its id.

    If the same sample already exists, the existing row id is returned.
    """

    if label not in (0, 1):
        raise ValueError("label must be 0 for normal mail or 1 for spam mail")

    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO training_data (content, label, source)
            VALUES (?, ?, ?)
            """,
            (content, label, source),
        )
        row = connection.execute(
            """
            SELECT id FROM training_data
            WHERE content = ? AND label = ? AND source = ?
            """,
            (content, label, source),
        ).fetchone()
        connection.commit()
        return int(row["id"])


def list_training_data(db_path: str | Path = DEFAULT_DB_PATH) -> list[sqlite3.Row]:
    """Return all training samples ordered by id."""

    with get_connection(db_path) as connection:
        return connection.execute("SELECT * FROM training_data ORDER BY id").fetchall()


def add_email(
    sender_id: int,
    receiver_id: int,
    subject: str,
    content: str,
    folder: str,
    spam_probability: float = 0.0,
    normal_probability: float = 0.0,
    is_spam: int = 0,
    classification_method: str | None = None,
    classification_reason: str | None = None,
    send_time: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Insert one email record and return its id."""

    if folder not in {"inbox", "sent", "trash"}:
        raise ValueError("folder must be one of: inbox, sent, trash")
    if is_spam not in (0, 1):
        raise ValueError("is_spam must be 0 or 1")

    timestamp = send_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as connection:
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
                timestamp,
                spam_probability,
                normal_probability,
                is_spam,
                folder,
                classification_method,
                classification_reason,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_emails(
    user_id: int | None = None,
    folder: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[sqlite3.Row]:
    """Return emails, optionally filtered by receiver/sender user and folder."""

    query = "SELECT * FROM emails"
    conditions: list[str] = []
    params: list[object] = []

    if user_id is not None:
        conditions.append("(receiver_id = ? OR sender_id = ?)")
        params.extend([user_id, user_id])
    if folder is not None:
        conditions.append("folder = ?")
        params.append(folder)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id DESC"

    with get_connection(db_path) as connection:
        return connection.execute(query, params).fetchall()


def get_table_counts(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, int]:
    """Return row counts for quick verification and demos."""

    with get_connection(db_path) as connection:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("users", "emails", "training_data")
        }


if __name__ == "__main__":
    initialize_database()
    counts = get_table_counts()
    print("数据库初始化完成")
    print(f"路径: {DEFAULT_DB_PATH}")
    print(f"用户数量: {counts['users']}")
    print(f"邮件数量: {counts['emails']}")
    print(f"训练样本数量: {counts['training_data']}")
