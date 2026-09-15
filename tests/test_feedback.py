import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import bayes
import database
import mail_service


class FeedbackLearningTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "mail.db"
        database.initialize_database(self.db_path)

        self.zhangsan = database.get_user_by_username("张三", self.db_path)
        self.lisi = database.get_user_by_username("李四", self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _feedback_rows(self, text: str | None = None):
        rows = [
            row
            for row in database.list_training_data(self.db_path)
            if row["source"] == "user_feedback"
        ]
        if text is not None:
            rows = [row for row in rows if row["content"] == text]
        return rows

    def test_mark_as_normal_moves_trash_mail_to_inbox_and_adds_feedback(self):
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "礼品提醒",
            "限时领取礼品，请点击链接查看奖励。",
            self.db_path,
        )
        received_id = result["received_email_id"]

        correction = mail_service.mark_as_normal(received_id, self.db_path)
        updated_email = database.get_email_by_id(received_id, self.db_path)
        feedback_rows = self._feedback_rows(correction["training_text"])

        self.assertEqual(correction["folder"], "inbox")
        self.assertFalse(correction["is_spam"])
        self.assertEqual(updated_email["folder"], "inbox")
        self.assertEqual(updated_email["is_spam"], 0)
        self.assertEqual(len(feedback_rows), 1)
        self.assertEqual(feedback_rows[0]["label"], 0)
        self.assertEqual(feedback_rows[0]["source"], "user_feedback")

    def test_mark_as_spam_moves_inbox_mail_to_trash_and_adds_feedback(self):
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "课程资料",
            "请点击链接查看课程资料。",
            self.db_path,
        )
        received_id = result["received_email_id"]

        correction = mail_service.mark_as_spam(received_id, self.db_path)
        updated_email = database.get_email_by_id(received_id, self.db_path)
        feedback_rows = self._feedback_rows(correction["training_text"])

        self.assertEqual(correction["folder"], "trash")
        self.assertTrue(correction["is_spam"])
        self.assertEqual(updated_email["folder"], "trash")
        self.assertEqual(updated_email["is_spam"], 1)
        self.assertEqual(len(feedback_rows), 1)
        self.assertEqual(feedback_rows[0]["label"], 1)
        self.assertEqual(feedback_rows[0]["source"], "user_feedback")

    def test_sender_sent_record_is_not_changed_by_receiver_correction(self):
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "礼品提醒",
            "限时领取礼品，请点击链接查看奖励。",
            self.db_path,
        )

        mail_service.mark_as_normal(result["received_email_id"], self.db_path)
        sent_email = database.get_email_by_id(result["sent_email_id"], self.db_path)

        self.assertEqual(sent_email["folder"], "sent")
        self.assertEqual(sent_email["is_spam"], 1)

    def test_sent_mail_cannot_be_corrected(self):
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "项目会议",
            "明天下午三点召开项目会议，请准时参加。",
            self.db_path,
        )

        with self.assertRaisesRegex(ValueError, "sent mail cannot be corrected"):
            mail_service.mark_as_spam(result["sent_email_id"], self.db_path)

    def test_repeating_same_feedback_does_not_duplicate_training_sample(self):
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "礼品提醒",
            "限时领取礼品，请点击链接查看奖励。",
            self.db_path,
        )
        received_id = result["received_email_id"]

        first = mail_service.mark_as_normal(received_id, self.db_path)
        second = mail_service.mark_as_normal(received_id, self.db_path)
        feedback_rows = self._feedback_rows(first["training_text"])

        self.assertEqual(first["training_sample_id"], second["training_sample_id"])
        self.assertEqual(len(feedback_rows), 1)
        self.assertEqual(feedback_rows[0]["label"], 0)

    def test_opposite_feedback_updates_existing_user_feedback_label(self):
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "礼品提醒",
            "限时领取礼品，请点击链接查看奖励。",
            self.db_path,
        )
        received_id = result["received_email_id"]

        normal = mail_service.mark_as_normal(received_id, self.db_path)
        spam = mail_service.mark_as_spam(received_id, self.db_path)
        feedback_rows = self._feedback_rows(normal["training_text"])
        updated_email = database.get_email_by_id(received_id, self.db_path)

        self.assertEqual(normal["training_sample_id"], spam["training_sample_id"])
        self.assertEqual(len(feedback_rows), 1)
        self.assertEqual(feedback_rows[0]["label"], 1)
        self.assertEqual(updated_email["folder"], "trash")
        self.assertEqual(updated_email["is_spam"], 1)

    def test_normal_feedback_lowers_future_spam_probability(self):
        subject = "礼品提醒"
        content = "限时领取礼品，请点击链接查看奖励。"
        text = mail_service.combine_mail_text(subject, content)
        before_spam_probability, _ = bayes.bayes_predict(text, self.db_path)
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            subject,
            content,
            self.db_path,
        )

        mail_service.mark_as_normal(result["received_email_id"], self.db_path)
        after_spam_probability, _ = bayes.bayes_predict(text, self.db_path)

        self.assertLess(after_spam_probability, before_spam_probability)

    def test_spam_feedback_raises_future_spam_probability(self):
        subject = "课程资料"
        content = "请点击链接查看课程资料。"
        text = mail_service.combine_mail_text(subject, content)
        before_spam_probability, _ = bayes.bayes_predict(text, self.db_path)
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            subject,
            content,
            self.db_path,
        )

        mail_service.mark_as_spam(result["received_email_id"], self.db_path)
        after_spam_probability, _ = bayes.bayes_predict(text, self.db_path)

        self.assertGreater(after_spam_probability, before_spam_probability)

    def test_hard_rule_still_wins_after_user_marks_current_mail_normal(self):
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "课程优惠说明",
            "这是学校课程的优惠信息。",
            self.db_path,
        )

        mail_service.mark_as_normal(result["received_email_id"], self.db_path)
        repeated = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "课程优惠说明",
            "这是学校课程的优惠信息。",
            self.db_path,
        )

        self.assertEqual(repeated["receiver_folder"], "trash")
        self.assertEqual(repeated["classification"]["classification_method"], "rule")


if __name__ == "__main__":
    unittest.main(verbosity=2)
