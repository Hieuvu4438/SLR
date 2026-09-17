import unittest

from .discourse_metadata_audit import adjacency, inferred_index


class DiscourseMetadataTests(unittest.TestCase):
    def test_last_hyphen_only(self):
        self.assertEqual(inferred_index('date-show-name-12'), ('date-show-name', 12))
        self.assertIsNone(inferred_index('date-show-name-last'))

    def test_missing_and_cross_split_are_not_continuity(self):
        result = adjacency({'train': ['a-1', 'a-2', 'a-4'], 'dev': ['a-3', 'b-2', 'bad']})
        self.assertEqual(result['train'], dict(parsed=3, unparsed=0, predecessor_same_split=1,
                                              predecessor_other_inspected_split=1,
                                              predecessor_absent_from_inspected_train_dev=1))
        self.assertEqual(result['dev'], dict(parsed=2, unparsed=1, predecessor_same_split=0,
                                            predecessor_other_inspected_split=1,
                                            predecessor_absent_from_inspected_train_dev=1))

    def test_ambiguous_numeric_keys_fail_closed(self):
        with self.assertRaises(ValueError):
            adjacency({'train': ['a-01'], 'dev': ['a-1']})


if __name__ == '__main__':
    unittest.main()
