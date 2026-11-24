"""Regression-test the prepared benchmark benchmark contract."""

import unittest

import numpy as np

from benchmark.data import MultiSourceWindowDataset, PreparedBenchmarkStream, SourceSeries


class PreparedBenchmarkStreamTests(unittest.TestCase):
    """Define the PreparedBenchmarkStreamTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    @staticmethod
    def _dataset() -> MultiSourceWindowDataset:
        """Construct a deterministic synthetic window dataset for prepared-stream tests.
        
        Returns:
            MultiSourceWindowDataset: Result produced by this operation.
        """
        source = SourceSeries(
            source_id="cpcb/site_1_PM2.5",
            source_file="site_1_PM2.5.csv",
            source_path="/dataset/cpcb/site_1_PM2.5.csv",
            source_relative_path="cpcb/site_1_PM2.5.csv",
            dataset_group="cpcb",
            target_column="PM2.5",
            time_column="Timestamp",
            values=np.arange(14, dtype=np.float32),
            observed_mask=None,
            timestamps_ns=None,
            regular_start_timestamp_ns=0,
            expected_interval_ns=1,
            segments=np.asarray([(0, 14)], dtype=np.int64),
            preprocessing={},
        )
        return MultiSourceWindowDataset(
            [source], context_len=4, horizon_len=2, stride=2, include_targets=False
        )

    def test_batches_are_preloaded_contiguous_views_with_deterministic_wrap(self):
        """Regression-check that batches are preloaded contiguous views with deterministic wrap.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        stream = PreparedBenchmarkStream(
            self._dataset(), batch_size=3, batch_count=4, max_pool_windows=10
        )

        self.assertEqual(stream.pool_windows, 5)
        self.assertTrue(stream.repeated)
        self.assertTrue(stream.batch(0).flags.c_contiguous)
        self.assertEqual(stream.batch(0).tolist(), [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]])
        self.assertEqual(stream.batch(1).tolist(), [[6, 7, 8, 9], [8, 9, 10, 11], [0, 1, 2, 3]])
        self.assertEqual(stream.batch(2).tolist(), [[2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]])

    def test_batch_larger_than_dataset_repeats_without_runtime_assembly(self):
        """Regression-check that batch larger than dataset repeats without runtime assembly.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        stream = PreparedBenchmarkStream(
            self._dataset(), batch_size=8, batch_count=2, max_pool_windows=2
        )

        self.assertEqual(stream.pool_windows, 5)
        self.assertEqual(stream.batch(0).shape, (8, 4))
        self.assertEqual(stream.batch(1).shape, (8, 4))
        self.assertTrue(np.shares_memory(stream.batch(0), stream.inputs))


if __name__ == "__main__":
    unittest.main()
