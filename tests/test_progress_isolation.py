"""Regression-test the progress isolation benchmark contract."""

import unittest
from unittest.mock import patch

from benchmark.runner import live_batch_progress_enabled


class ProgressIsolationTests(unittest.TestCase):
    """Define the ProgressIsolationTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def test_measured_iterations_never_enable_live_batch_redraws(self):
        """Regression-check that measured iterations never enable live batch redraws.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with patch("benchmark.runner.PROGRESS_DISABLED", False):
            self.assertFalse(live_batch_progress_enabled(is_warmup=False))
            self.assertTrue(live_batch_progress_enabled(is_warmup=True))

if __name__ == "__main__":
    unittest.main()
