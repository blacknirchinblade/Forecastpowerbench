import os
import unittest
from unittest.mock import Mock, patch, sentinel

from benchmark.runner import BenchmarkRunner


TRANSIENT_ERROR = TypeError(
    "unsupported operand type(s) for /: 'cell' and 'bool'"
)


class MomentModelLoadRetryTests(unittest.TestCase):
    def setUp(self):
        self.runner = object.__new__(BenchmarkRunner)
        self.runner.benchmark_seed = 12345
        self.arguments = {
            "model_name": "moment",
            "device": "cuda",
            "context_len": 128,
            "horizon_len": 1,
            "batch_size": 512,
            "num_workers": 0,
        }

    @patch.dict(
        os.environ,
        {
            "MOMENT_MODEL_LOAD_RETRIES": "2",
            "MOMENT_MODEL_LOAD_RETRY_DELAY_SEC": "0",
        },
    )
    @patch("benchmark.runner.seed_benchmark_rng")
    @patch("benchmark.runner.clear_gpu_memory")
    @patch("benchmark.runner.get_model")
    def test_retries_exact_transient_moment_failure(
        self,
        get_model_mock: Mock,
        clear_gpu_memory_mock: Mock,
        seed_mock: Mock,
    ):
        get_model_mock.side_effect = [TRANSIENT_ERROR, sentinel.model]

        result = self.runner._load_model_for_config(**self.arguments)

        self.assertIs(result, sentinel.model)
        self.assertEqual(get_model_mock.call_count, 2)
        self.assertEqual(seed_mock.call_count, 2)
        clear_gpu_memory_mock.assert_called_once_with()

    @patch.dict(
        os.environ,
        {
            "MOMENT_MODEL_LOAD_RETRIES": "2",
            "MOMENT_MODEL_LOAD_RETRY_DELAY_SEC": "0",
        },
    )
    @patch("benchmark.runner.seed_benchmark_rng")
    @patch("benchmark.runner.clear_gpu_memory")
    @patch("benchmark.runner.get_model")
    def test_does_not_hide_unrelated_type_error(
        self,
        get_model_mock: Mock,
        clear_gpu_memory_mock: Mock,
        _seed_mock: Mock,
    ):
        get_model_mock.side_effect = TypeError("invalid adapter configuration")

        with self.assertRaisesRegex(TypeError, "invalid adapter configuration"):
            self.runner._load_model_for_config(**self.arguments)

        get_model_mock.assert_called_once()
        clear_gpu_memory_mock.assert_not_called()

    @patch.dict(
        os.environ,
        {
            "MOMENT_MODEL_LOAD_RETRIES": "2",
            "MOMENT_MODEL_LOAD_RETRY_DELAY_SEC": "0",
        },
    )
    @patch("benchmark.runner.seed_benchmark_rng")
    @patch("benchmark.runner.clear_gpu_memory")
    @patch("benchmark.runner.get_model")
    def test_does_not_retry_other_models(
        self,
        get_model_mock: Mock,
        clear_gpu_memory_mock: Mock,
        _seed_mock: Mock,
    ):
        get_model_mock.side_effect = TRANSIENT_ERROR
        arguments = {**self.arguments, "model_name": "units"}

        with self.assertRaisesRegex(TypeError, "'cell' and 'bool'"):
            self.runner._load_model_for_config(**arguments)

        get_model_mock.assert_called_once()
        clear_gpu_memory_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
