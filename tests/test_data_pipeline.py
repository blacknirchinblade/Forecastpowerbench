"""Regression-test the data pipeline benchmark contract."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from benchmark.data import (
    MultiSourceWindowDataset,
    discover_source_files,
    identity_window_batch,
    load_source_directory,
)


class MultiSourceDataPipelineTests(unittest.TestCase):
    """Define the MultiSourceDataPipelineTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    @staticmethod
    def _write_source(path: Path, values) -> None:
        """Write source for MultiSourceDataPipelineTests.
        
        Args:
            path (Path): Filesystem path consumed by the operation.
            values (Any): Values to normalize, summarize, or serialize.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Timestamp", "PM2.5 (ug/m3)"])
            for index, value in enumerate(values):
                writer.writerow([f"2024-01-01 {index:02d}:00:00", value])

    def test_recursive_sources_are_windowed_independently_in_stable_order(self):
        """Regression-check that recursive sources are windowed independently in stable order.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_source(root / "group_a" / "same.csv", range(12))
            self._write_source(root / "group_b" / "same.csv", range(100, 108))

            cache_dir = root / ".cache"
            sources = load_source_directory(root, cache_dir=cache_dir)
            self.assertEqual(
                [source.source_id for source in sources],
                ["group_a/same", "group_b/same"],
            )
            self.assertTrue(all(source.timestamps_ns is None for source in sources))
            self.assertTrue(
                all(source.regular_start_timestamp_ns is not None for source in sources)
            )
            self.assertTrue(all(source.observed_mask is None for source in sources))

            dataset = MultiSourceWindowDataset(sources, context_len=4, horizon_len=3, stride=3)
            self.assertEqual(dataset.windows_by_source, {"group_a/same": 2, "group_b/same": 1})
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.source_indices.tolist(), [0, 0, 1])
            self.assertEqual(dataset.starts.tolist(), [0, 3, 0])

            first_context, first_target, source_index, window_id, start, _ = dataset[0]
            self.assertEqual(first_context.tolist(), [0.0, 1.0, 2.0, 3.0])
            self.assertEqual(first_target.tolist(), [4.0, 5.0, 6.0])
            self.assertEqual((source_index, window_id, start), (0, 0, 0))

            last_context, last_target, source_index, window_id, start, _ = dataset[2]
            self.assertEqual(last_context.tolist(), [100.0, 101.0, 102.0, 103.0])
            self.assertEqual(last_target.tolist(), [104.0, 105.0, 106.0])
            self.assertEqual((source_index, window_id, start), (1, 0, 0))

            vectorized = MultiSourceWindowDataset(
                sources,
                context_len=4,
                horizon_len=3,
                stride=3,
                include_targets=False,
            )
            loader = DataLoader(
                vectorized,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                collate_fn=identity_window_batch,
            )
            batch_x, batch_y, batch_sources, batch_windows, batch_starts, batch_indices = next(
                iter(loader)
            )
            self.assertEqual(tuple(batch_x.shape), (2, 4))
            self.assertEqual(tuple(batch_y.shape), (2, 0))
            self.assertEqual(batch_x.tolist(), [[0.0, 1.0, 2.0, 3.0], [3.0, 4.0, 5.0, 6.0]])
            self.assertEqual(batch_sources.tolist(), [0, 0])
            self.assertEqual(batch_windows.tolist(), [0, 1])
            self.assertEqual(batch_starts.tolist(), [0, 3])
            self.assertEqual(batch_indices.tolist(), [0, 1])

            cached_sources = load_source_directory(root, cache_dir=cache_dir)
            self.assertEqual(
                [source.values.tolist() for source in cached_sources],
                [source.values.tolist() for source in sources],
            )
            self.assertEqual(len(list(cache_dir.rglob("*.npz"))), 2)

    def test_missing_targets_are_median_imputed_but_remain_masked(self):
        """Regression-check that missing targets are median imputed but remain masked.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_source(root / "group" / "missing.csv", [1, 2, "", 4, 5, 6, 7])
            source = load_source_directory(root)[0]

            self.assertEqual(source.values.dtype, np.float32)
            self.assertEqual(source.values.tolist(), [1.0, 2.0, 4.5, 4.0, 5.0, 6.0, 7.0])
            self.assertIsNotNone(source.observed_mask)
            self.assertEqual(
                source.observed_mask.tolist(), [True, True, False, True, True, True, True]
            )
            self.assertEqual(source.preprocessing["target_values_imputed"], 1)

    def test_dataset_folder_and_pollutant_pattern_are_replaceable(self):
        """Regression-check that dataset folder and pollutant pattern are replaceable.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_source(root / "future_dataset" / "region" / "site_PM2.5.csv", range(8))
            self._write_source(root / "future_dataset" / "region" / "site_NO2.csv", range(8))

            dataset_folder = "future_dataset"
            pollutant = "PM2.5"
            pattern = f"{dataset_folder}/**/*_{pollutant}.csv"
            matches = discover_source_files(root, pattern=pattern)

            self.assertEqual(
                [path.relative_to(root).as_posix() for path in matches],
                ["future_dataset/region/site_PM2.5.csv"],
            )


if __name__ == "__main__":
    unittest.main()
