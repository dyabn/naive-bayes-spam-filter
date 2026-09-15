import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import database
import mail_service


class MailClassificationServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "mail.db"
        database.initialize_database(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_subject_hard_keyword_marks_spam_by_rule(self):
        result = mail_service.classify_mail("恭喜中奖", "请及时查看", self.db_path)

        self.assertTrue(result["is_spam"])
        self.assertEqual(result["classification_method"], "rule")
        self.assertIsNone(result["spam_probability"])
        self.assertIsNone(result["normal_probability"])
        self.assertEqual(result["matched_keywords"], ["中奖"])

    def test_content_hard_keyword_marks_spam_by_rule(self):
        result = mail_service.classify_mail("活动通知", "今日优惠活动开始", self.db_path)

        self.assertTrue(result["is_spam"])
        self.assertEqual(result["classification_method"], "rule")
        self.assertEqual(result["matched_keywords"], ["优惠"])

    def test_multiple_hard_keywords_are_recorded_in_configured_order(self):
        result = mail_service.classify_mail("中奖优惠活动", "请查看", self.db_path)

        self.assertEqual(result["matched_keywords"], ["优惠", "中奖"])
        self.assertIn("优惠", result["classification_reason"])
        self.assertIn("中奖", result["classification_reason"])

    def test_hard_rule_has_priority_over_bayes(self):
        result = mail_service.classify_mail("项目会议优惠说明", "明天下午三点召开会议", self.db_path)

        self.assertTrue(result["is_spam"])
        self.assertEqual(result["classification_method"], "rule")
        self.assertIsNone(result["spam_probability"])

    def test_bayes_normal_mail_is_not_spam(self):
        result = mail_service.classify_mail(
            "会议通知",
            "明天下午三点召开项目会议，请准时参加。",
            self.db_path,
        )

        self.assertFalse(result["is_spam"])
        self.assertEqual(result["classification_method"], "bayes")
        self.assertGreater(result["normal_probability"], result["spam_probability"])

    def test_bayes_spam_mail_without_hard_words_is_spam(self):
        result = mail_service.classify_mail(
            "礼品提醒",
            "限时领取礼品，请点击链接查看奖励。",
            self.db_path,
        )

        self.assertTrue(result["is_spam"])
        self.assertEqual(result["classification_method"], "bayes")
        self.assertGreater(result["spam_probability"], 0.55)
        self.assertGreater(result["spam_probability"], result["normal_probability"])
        self.assertEqual(
            result["matched_keywords"],
            ["限时", "领取", "点击", "链接", "礼品", "奖励"],
        )

    def test_empty_subject_does_not_crash(self):
        result = mail_service.classify_mail("", "明天下午召开会议", self.db_path)

        self.assertEqual(result["classification_method"], "bayes")
        self.assertIsInstance(result["is_spam"], bool)

    def test_empty_subject_and_content_do_not_crash(self):
        result = mail_service.classify_mail("", "", self.db_path)

        self.assertEqual(result["classification_method"], "bayes")
        self.assertIsNotNone(result["spam_probability"])
        self.assertIsNotNone(result["normal_probability"])
        self.assertAlmostEqual(
            result["spam_probability"] + result["normal_probability"],
            1.0,
        )

    def test_matched_feature_words_can_be_used_by_ui(self):
        words = mail_service.get_matched_feature_words("限时领取礼品，请点击链接查看奖励")

        self.assertEqual(words, ["限时", "领取", "点击", "链接", "礼品", "奖励"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
