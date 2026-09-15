import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import database
import mail_service


class MailFlowWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "mail.db"
        database.initialize_database(self.db_path)

        self.zhangsan = database.get_user_by_username("张三", self.db_path)
        self.lisi = database.get_user_by_username("李四", self.db_path)
        self.wangwu = database.get_user_by_username("王五", self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_send_normal_mail_creates_sent_and_inbox_records(self):
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "项目会议",
            "明天下午三点召开项目会议，请准时参加。",
            self.db_path,
        )

        self.assertEqual(result["receiver_folder"], "inbox")
        self.assertNotEqual(result["sent_email_id"], result["received_email_id"])

        sent = mail_service.get_sent(self.zhangsan["id"], self.db_path)
        inbox = mail_service.get_inbox(self.lisi["id"], self.db_path)
        trash = mail_service.get_trash(self.lisi["id"], self.db_path)

        self.assertEqual(len(sent), 1)
        self.assertEqual(len(inbox), 1)
        self.assertEqual(len(trash), 0)
        self.assertEqual(sent[0]["folder"], "sent")
        self.assertEqual(inbox[0]["folder"], "inbox")
        self.assertEqual(inbox[0]["classification_method"], "bayes")
        self.assertEqual(inbox[0]["is_spam"], 0)

    def test_hard_rule_mail_creates_sent_and_trash_records_with_null_probabilities(self):
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "活动通知",
            "恭喜中奖，请及时领取。",
            self.db_path,
        )

        self.assertEqual(result["receiver_folder"], "trash")
        self.assertEqual(result["classification"]["classification_method"], "rule")
        self.assertIsNone(result["classification"]["spam_probability"])

        sent = mail_service.get_sent(self.zhangsan["id"], self.db_path)
        trash = mail_service.get_trash(self.lisi["id"], self.db_path)

        self.assertEqual(len(sent), 1)
        self.assertEqual(len(trash), 1)
        self.assertEqual(trash[0]["classification_method"], "rule")
        self.assertIsNone(trash[0]["spam_probability"])
        self.assertIsNone(trash[0]["normal_probability"])
        self.assertEqual(trash[0]["is_spam"], 1)

    def test_bayes_spam_without_hard_words_goes_to_trash(self):
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "礼品通知",
            "限时领取礼品，请点击链接查看奖励。",
            self.db_path,
        )

        self.assertEqual(result["receiver_folder"], "trash")
        self.assertEqual(result["classification"]["classification_method"], "bayes")
        self.assertGreater(result["classification"]["spam_probability"], 0.55)

        trash = mail_service.get_trash(self.lisi["id"], self.db_path)
        self.assertEqual(len(trash), 1)
        self.assertEqual(trash[0]["classification_method"], "bayes")
        self.assertGreater(trash[0]["spam_probability"], 0.55)

    def test_inbox_does_not_include_sender_sent_mail_or_trash_mail(self):
        mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "项目会议",
            "明天下午三点召开项目会议，请准时参加。",
            self.db_path,
        )
        mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "活动通知",
            "恭喜中奖，请及时领取。",
            self.db_path,
        )

        inbox = mail_service.get_inbox(self.lisi["id"], self.db_path)

        self.assertEqual(len(inbox), 1)
        self.assertEqual(inbox[0]["subject"], "项目会议")
        self.assertEqual(inbox[0]["receiver_id"], self.lisi["id"])
        self.assertEqual(inbox[0]["folder"], "inbox")

    def test_sent_returns_only_user_sent_records(self):
        mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "张三邮件",
            "明天下午三点召开项目会议，请准时参加。",
            self.db_path,
        )
        mail_service.send_mail(
            self.lisi["id"],
            self.zhangsan["id"],
            "李四邮件",
            "请查看课堂资料。",
            self.db_path,
        )

        zhangsan_sent = mail_service.get_sent(self.zhangsan["id"], self.db_path)

        self.assertEqual(len(zhangsan_sent), 1)
        self.assertEqual(zhangsan_sent[0]["subject"], "张三邮件")
        self.assertEqual(zhangsan_sent[0]["sender_id"], self.zhangsan["id"])
        self.assertEqual(zhangsan_sent[0]["folder"], "sent")

    def test_trash_returns_only_user_received_spam_records(self):
        mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "李四垃圾邮件",
            "恭喜中奖，请及时领取。",
            self.db_path,
        )
        mail_service.send_mail(
            self.lisi["id"],
            self.zhangsan["id"],
            "张三垃圾邮件",
            "今日优惠活动开始。",
            self.db_path,
        )

        lisi_trash = mail_service.get_trash(self.lisi["id"], self.db_path)

        self.assertEqual(len(lisi_trash), 1)
        self.assertEqual(lisi_trash[0]["subject"], "李四垃圾邮件")
        self.assertEqual(lisi_trash[0]["receiver_id"], self.lisi["id"])
        self.assertEqual(lisi_trash[0]["folder"], "trash")

    def test_multiple_sends_have_correct_folder_counts(self):
        mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "项目会议",
            "明天下午三点召开项目会议，请准时参加。",
            self.db_path,
        )
        mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "活动通知",
            "恭喜中奖，请及时领取。",
            self.db_path,
        )
        mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "礼品通知",
            "限时领取礼品，请点击链接查看奖励。",
            self.db_path,
        )

        self.assertEqual(len(mail_service.get_sent(self.zhangsan["id"], self.db_path)), 3)
        self.assertEqual(len(mail_service.get_inbox(self.lisi["id"], self.db_path)), 1)
        self.assertEqual(len(mail_service.get_trash(self.lisi["id"], self.db_path)), 2)

    def test_user_without_mail_gets_empty_lists(self):
        self.assertEqual(mail_service.get_sent(self.wangwu["id"], self.db_path), [])
        self.assertEqual(mail_service.get_inbox(self.wangwu["id"], self.db_path), [])
        self.assertEqual(mail_service.get_trash(self.wangwu["id"], self.db_path), [])

    def test_received_record_matches_classification_result(self):
        result = mail_service.send_mail(
            self.zhangsan["id"],
            self.lisi["id"],
            "礼品通知",
            "限时领取礼品，请点击链接查看奖励。",
            self.db_path,
        )

        received_email = mail_service.get_trash(self.lisi["id"], self.db_path)[0]
        classification = result["classification"]

        self.assertAlmostEqual(
            received_email["spam_probability"],
            classification["spam_probability"],
        )
        self.assertAlmostEqual(
            received_email["normal_probability"],
            classification["normal_probability"],
        )
        self.assertEqual(
            received_email["classification_method"],
            classification["classification_method"],
        )
        self.assertEqual(
            received_email["classification_reason"],
            classification["classification_reason"],
        )

    def test_send_mail_rolls_back_when_receiver_does_not_exist(self):
        with self.assertRaises(sqlite3.IntegrityError):
            mail_service.send_mail(
                self.zhangsan["id"],
                9999,
                "无效收件人",
                "明天下午三点召开项目会议，请准时参加。",
                self.db_path,
            )

        self.assertEqual(mail_service.get_sent(self.zhangsan["id"], self.db_path), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
