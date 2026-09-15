import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import database


class DatabaseInitializationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "mail.db"
        database.initialize_database(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_required_tables_are_created(self):
        with database.get_connection(self.db_path) as connection:
            table_names = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }

        self.assertIn("users", table_names)
        self.assertIn("emails", table_names)
        self.assertIn("training_data", table_names)

    def test_demo_users_are_seeded(self):
        users = database.list_users(self.db_path)
        usernames = {user["username"] for user in users}

        self.assertIn("张三", usernames)
        self.assertIn("李四", usernames)
        self.assertIn("王五", usernames)

    def test_training_data_contains_normal_and_spam_samples(self):
        samples = database.list_training_data(self.db_path)
        labels = [sample["label"] for sample in samples]

        self.assertGreaterEqual(labels.count(0), 6)
        self.assertGreaterEqual(labels.count(1), 6)

    def test_emails_table_has_frozen_design_fields(self):
        expected_columns = {
            "id",
            "sender_id",
            "receiver_id",
            "subject",
            "content",
            "send_time",
            "spam_probability",
            "normal_probability",
            "is_spam",
            "folder",
            "classification_method",
            "classification_reason",
        }

        with database.get_connection(self.db_path) as connection:
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(emails)")
            }

        self.assertTrue(expected_columns.issubset(columns))

    def test_email_probability_columns_are_nullable_after_migration(self):
        legacy_db_path = Path(self.temp_dir.name) / "legacy.db"
        with database.get_connection(legacy_db_path) as connection:
            connection.executescript(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT
                );

                CREATE TABLE emails (
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

                CREATE TABLE training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    label INTEGER NOT NULL CHECK (label IN (0, 1)),
                    source TEXT NOT NULL DEFAULT 'initial',
                    UNIQUE (content, label, source)
                );

                INSERT INTO users (username, password) VALUES ('张三', '123456');
                INSERT INTO users (username, password) VALUES ('李四', '123456');
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
                VALUES (
                    1,
                    2,
                    '旧邮件',
                    '旧版数据库邮件',
                    '2026-09-15 10:00:00',
                    0.0,
                    1.0,
                    0,
                    'inbox',
                    'bayes',
                    '旧记录'
                );
                """
            )
            connection.commit()
            database.create_tables(connection)

            columns = {
                row["name"]: row
                for row in connection.execute("PRAGMA table_info(emails)")
            }
            row_count = connection.execute("SELECT COUNT(*) FROM emails").fetchone()[0]

        self.assertEqual(columns["spam_probability"]["notnull"], 0)
        self.assertEqual(columns["normal_probability"]["notnull"], 0)
        self.assertEqual(row_count, 1)

    def test_initialization_is_idempotent(self):
        before = database.get_table_counts(self.db_path)
        database.initialize_database(self.db_path)
        after = database.get_table_counts(self.db_path)

        self.assertEqual(before, after)

    def test_can_insert_email_record(self):
        sender = database.get_user_by_username("张三", self.db_path)
        receiver = database.get_user_by_username("李四", self.db_path)

        email_id = database.add_email(
            sender_id=sender["id"],
            receiver_id=receiver["id"],
            subject="会议通知",
            content="明天下午三点召开项目会议，请准时参加。",
            folder="inbox",
            spam_probability=0.12,
            normal_probability=0.88,
            is_spam=0,
            classification_method="bayes",
            classification_reason="正常邮件概率较高",
            db_path=self.db_path,
        )

        self.assertIsInstance(email_id, int)
        emails = database.list_emails(receiver["id"], "inbox", self.db_path)
        self.assertEqual(len(emails), 1)
        self.assertEqual(emails[0]["subject"], "会议通知")

    def test_rejects_invalid_training_label(self):
        with self.assertRaises(ValueError):
            database.add_training_sample("invalid", 2, db_path=self.db_path)

    def test_foreign_keys_are_enforced(self):
        with self.assertRaises(sqlite3.IntegrityError):
            database.add_email(
                sender_id=999,
                receiver_id=1000,
                subject="无效邮件",
                content="用户不存在",
                folder="inbox",
                db_path=self.db_path,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
