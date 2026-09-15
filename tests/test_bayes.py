import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import bayes
import database


class BernoulliNaiveBayesTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "mail.db"
        database.initialize_database(self.db_path)
        self.samples = bayes.load_training_samples(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_features_converts_keywords_to_binary_values(self):
        features = bayes.extract_features("限时领取礼品，请点击链接")

        self.assertEqual(features["限时"], 1)
        self.assertEqual(features["领取"], 1)
        self.assertEqual(features["点击"], 1)
        self.assertEqual(features["链接"], 1)
        self.assertEqual(features["礼品"], 1)
        self.assertEqual(features["现金"], 0)
        self.assertEqual(set(features), set(bayes.FEATURE_WORDS))

    def test_repeated_keyword_still_counts_as_one(self):
        features = bayes.extract_features("中奖中奖中奖")

        self.assertEqual(features["中奖"], 1)

    def test_priors_are_balanced_for_initial_training_data(self):
        priors = bayes.calculate_priors(self.samples)

        self.assertAlmostEqual(priors[bayes.SPAM_LABEL], 0.5)
        self.assertAlmostEqual(priors[bayes.NORMAL_LABEL], 0.5)

    def test_conditional_probabilities_are_laplace_smoothed(self):
        probabilities = bayes.calculate_conditional_probabilities(self.samples)

        for label_probabilities in probabilities.values():
            for probability in label_probabilities.values():
                self.assertGreater(probability, 0)
                self.assertLess(probability, 1)

    def test_normal_text_has_higher_normal_probability(self):
        spam_probability, normal_probability = bayes.bayes_predict(
            "明天下午三点召开项目会议，请准时参加。",
            self.db_path,
        )

        self.assertGreater(normal_probability, spam_probability)

    def test_spam_like_text_without_hard_words_has_higher_spam_probability(self):
        spam_probability, normal_probability = bayes.bayes_predict(
            "限时领取礼品，请点击链接查看奖励。",
            self.db_path,
        )

        self.assertGreater(spam_probability, normal_probability)
        self.assertTrue(bayes.is_spam_probability(spam_probability))

    def test_probabilities_are_normalized(self):
        spam_probability, normal_probability = bayes.bayes_predict(
            "请点击链接查看课程资料。",
            self.db_path,
        )

        self.assertAlmostEqual(spam_probability + normal_probability, 1.0)

    def test_empty_text_does_not_crash(self):
        spam_probability, normal_probability = bayes.bayes_predict("", self.db_path)

        self.assertGreaterEqual(spam_probability, 0)
        self.assertGreaterEqual(normal_probability, 0)
        self.assertAlmostEqual(spam_probability + normal_probability, 1.0)

    def test_empty_training_data_raises_clear_error(self):
        empty_db_path = Path(self.temp_dir.name) / "empty.db"
        with database.get_connection(empty_db_path) as connection:
            database.create_tables(connection)

        with self.assertRaisesRegex(ValueError, "training samples are empty"):
            bayes.bayes_predict("测试文本", empty_db_path)

    def test_training_data_must_include_both_labels(self):
        one_label_db_path = Path(self.temp_dir.name) / "one-label.db"
        with database.get_connection(one_label_db_path) as connection:
            database.create_tables(connection)
        database.add_training_sample(
            "只有正常样本",
            bayes.NORMAL_LABEL,
            source="initial",
            db_path=one_label_db_path,
        )

        with self.assertRaisesRegex(ValueError, "both normal and spam labels"):
            bayes.bayes_predict("测试文本", one_label_db_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
