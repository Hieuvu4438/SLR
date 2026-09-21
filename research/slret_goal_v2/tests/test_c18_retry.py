import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_c18_batch64_v4 as pilot
import run_c18_batch64_retry_v4 as retry


class RetryTests(unittest.TestCase):
    def test_only_run_identity_and_telemetry_deadline_change(self):
        runner = pilot.runner
        original = vars(runner).copy()
        original_run = pilot.RUN
        try:
            pilot.configure()
            command = list(runner.COMMAND_OVERRIDE)
            configured = {key:getattr(runner,key) for key in (
                'MODE','REFERENCE','TOTAL_UNITS','HARD_SECONDS','OUTER_SECONDS',
                'ESTIMATED_SECONDS','validate')}
            retry.configure()
            command[command.index('--run-id')+1] = pilot.RUN
            self.assertEqual(command,runner.COMMAND_OVERRIDE)
            self.assertEqual(pilot.RUN,'seds-native-batch64-offload-002')
            self.assertEqual(runner.JOB,'v4-c18-batch64-002')
            self.assertEqual(runner.OUT.name,runner.JOB)
            self.assertEqual(runner.GPU_QUERY_TIMEOUT_SECONDS,20)
            self.assertEqual(configured,{k:getattr(runner,k) for k in configured})
        finally:
            vars(runner).update(original)
            pilot.RUN = original_run


if __name__ == '__main__':unittest.main()
