import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from rgb_bridge_state import ReplayState


class BridgeTests(unittest.TestCase):
    def test_two_commits(self):
        state=ReplayState()
        for version in range(2):
            state.encoded(version)
            state.backward_ready(version)
            state.committed(version)
        self.assertEqual((state.version,state.phase),(2,'idle'))

    def test_duplicate_stale_and_premature_commit_rejected(self):
        state=ReplayState()
        with self.assertRaises(ValueError): state.committed(0)
        state.encoded(0)
        with self.assertRaises(ValueError): state.encoded(0)
        with self.assertRaises(ValueError): state.backward_ready(1)
        state.backward_ready(0)
        state.committed(0)
        with self.assertRaises(ValueError): state.encoded(0)

    def test_no_version_advance_before_commit(self):
        state=ReplayState()
        state.encoded(0)
        state.backward_ready(0)
        self.assertEqual(state.version,0)


if __name__=='__main__': unittest.main()
