import sys
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from job_resources import ProcessTree


class ResourceTests(unittest.TestCase):
    def test_descendants_independent_of_process_group(self):
        table = {100:(1,10),101:(100,11),102:(101,12),900:(1,90)}
        with patch('job_resources.process_table',return_value=table),patch('job_resources.subprocess.check_output',return_value='102, 8000\n900, 9000\n'):
            tree = ProcessTree(100)
            self.assertEqual(tree.members(),{100,101,102})
            self.assertEqual(tree.gpu_bytes(),8000*1024**2)
            with patch('job_resources.os.kill') as kill:
                tree.send_signal(15)
                self.assertEqual({call.args[0] for call in kill.call_args_list},{100,101,102})
            table[102]=(1,12)  # Known orphan still owned.
            self.assertIn(102,tree.members())
            table[102]=(1,999)  # Reused PID no longer owned.
            self.assertNotIn(102,tree.members())

    def test_unknown_gpu_memory_fails_closed(self):
        with patch('job_resources.process_table',return_value={100:(1,10)}),patch('job_resources.subprocess.check_output',return_value='100, [N/A]\n'):
            with self.assertRaises(ValueError):ProcessTree(100).gpu_bytes()

    def test_query_deadline_is_configurable(self):
        with patch('job_resources.process_table',return_value={100:(1,10)}),patch(
                'job_resources.subprocess.check_output',return_value='100, 8000\n') as query:
            self.assertEqual(ProcessTree(100).gpu_bytes(timeout=20),8000*1024**2)
            self.assertEqual(query.call_args.kwargs['timeout'],20)

    def test_query_timeout_fails_closed_without_retry(self):
        with patch('job_resources.process_table',return_value={100:(1,10)}),patch(
                'job_resources.subprocess.check_output',
                side_effect=subprocess.TimeoutExpired('nvidia-smi',20)) as query:
            with self.assertRaises(subprocess.TimeoutExpired):
                ProcessTree(100).gpu_bytes(timeout=20)
            self.assertEqual(query.call_count,1)


if __name__=='__main__':unittest.main()
