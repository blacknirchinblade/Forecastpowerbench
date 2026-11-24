"""Regression tests for publication multi-objective analyses."""

import unittest

import pandas as pd

from analysis.publication import (
    alpha_score,
    multiobjective_alpha_score,
    normalize_multiobjective_alpha_components,
    normalize_score_components,
    robust_rank_acceptability,
)


class AlphaScoreTests(unittest.TestCase):
    """Verify the declared interpretation of the EAS alpha endpoints."""

    @staticmethod
    def _rows() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "nrmse": [1.0, 2.0, 3.0],
                "energy_j_per_forecast": [1.0, 10.0, 100.0],
            }
        )

    def test_alpha_endpoints_select_declared_component(self):
        rows, _bounds = normalize_score_components(
            self._rows(), error_metric="nrmse"
        )
        pd.testing.assert_series_equal(
            alpha_score(rows, 1.0),
            1.0 - rows["error_norm"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            alpha_score(rows, 0.0),
            1.0 - rows["energy_norm"],
            check_names=False,
        )

    def test_equal_alpha_averages_normalized_costs(self):
        rows, _bounds = normalize_score_components(
            self._rows(), error_metric="nrmse"
        )
        expected = 1.0 - 0.5 * (rows["error_norm"] + rows["energy_norm"])
        pd.testing.assert_series_equal(
            alpha_score(rows, 0.5), expected, check_names=False
        )

    def test_alpha_outside_unit_interval_is_rejected(self):
        rows, _bounds = normalize_score_components(
            self._rows(), error_metric="nrmse"
        )
        with self.assertRaises(ValueError):
            alpha_score(rows, -0.01)
        with self.assertRaises(ValueError):
            alpha_score(rows, 1.01)


class MultiObjectiveAlphaScoreTests(unittest.TestCase):
    """Verify fixed-priority four-objective alpha-score behavior."""

    @staticmethod
    def _rows() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "hardware": ["5090", "5090", "5090"],
                "nrmse": [1.0, 2.0, 3.0],
                "energy_j_per_forecast": [100.0, 10.0, 1.0],
                "inference_ms_per_forecast": [10.0, 5.0, 1.0],
                "memory_gb": [1.0, 2.0, 4.0],
            }
        )

    def test_independent_priorities_select_each_component(self):
        rows, _bounds = normalize_multiobjective_alpha_components(
            self._rows(), error_metric="nrmse"
        )
        for selected in ("error", "energy", "latency", "memory"):
            priorities = {
                criterion: float(criterion == selected)
                for criterion in ("error", "energy", "latency", "memory")
            }
            expected = 1.0 - rows[f"{selected}_deployment_norm"]
            pd.testing.assert_series_equal(
                multiobjective_alpha_score(rows, priorities),
                expected,
                check_names=False,
            )

    def test_common_priority_scale_does_not_change_score(self):
        rows, _bounds = normalize_multiobjective_alpha_components(
            self._rows(), error_metric="nrmse"
        )
        priorities = {"error": 4.0, "energy": 2.0, "latency": 1.0, "memory": 1.0}
        scaled = {name: value * 100.0 for name, value in priorities.items()}
        pd.testing.assert_series_equal(
            multiobjective_alpha_score(rows, priorities),
            multiobjective_alpha_score(rows, scaled),
        )

    def test_invalid_priority_vectors_are_rejected(self):
        rows, _bounds = normalize_multiobjective_alpha_components(
            self._rows(), error_metric="nrmse"
        )
        with self.assertRaises(ValueError):
            multiobjective_alpha_score(
                rows,
                {"error": 0.0, "energy": 0.0, "latency": 0.0, "memory": 0.0},
            )
        with self.assertRaises(ValueError):
            multiobjective_alpha_score(
                rows,
                {"error": 1.0, "energy": 1.0, "latency": -1.0, "memory": 1.0},
            )


class RobustRankingTests(unittest.TestCase):
    """Verify deterministic, complete-case, rank-based behavior."""

    @staticmethod
    def _rows(scale: float = 1.0) -> pd.DataFrame:
        records = []
        values = {
            "a": (1.0, 3.0, 3.0, 3.0),
            "b": (2.0, 2.0, 2.0, 2.0),
            "c": (3.0, 1.0, 1.0, 1.0),
        }
        for hardware in ("5090", "cpu"):
            for model, criteria in values.items():
                records.append(
                    {
                        "hardware": hardware,
                        "model_name": model,
                        "nrmse": criteria[0] * scale,
                        "energy_j_per_forecast": criteria[1] * scale,
                        "inference_ms_per_forecast": criteria[2] * scale,
                        "memory_gb": criteria[3] * scale,
                    }
                )
        return pd.DataFrame(records)

    def test_fixed_seed_is_reproducible(self):
        first = robust_rank_acceptability(
            self._rows(), error_metric="nrmse", n_draws=2_000, seed=12_345
        )
        second = robust_rank_acceptability(
            self._rows(), error_metric="nrmse", n_draws=2_000, seed=12_345
        )
        for first_frame, second_frame in zip(first, second):
            pd.testing.assert_frame_equal(first_frame, second_frame)

    def test_rank_score_is_invariant_to_common_positive_scale(self):
        base = robust_rank_acceptability(
            self._rows(), error_metric="nrmse", n_draws=2_000, seed=12_345
        )
        scaled = robust_rank_acceptability(
            self._rows(scale=1_000.0),
            error_metric="nrmse",
            n_draws=2_000,
            seed=12_345,
        )
        pd.testing.assert_frame_equal(base[0], scaled[0])
        pd.testing.assert_frame_equal(base[1], scaled[1])

    def test_incomplete_models_are_excluded(self):
        rows = self._rows()
        rows = rows[
            ~(
                rows["hardware"].eq("cpu")
                & rows["model_name"].eq("c")
            )
        ]
        per_hardware, aggregate, _costs = robust_rank_acceptability(
            rows, error_metric="nrmse", n_draws=500, seed=12_345
        )
        self.assertEqual(set(aggregate["model_name"]), {"a", "b"})
        self.assertEqual(set(per_hardware["model_name"]), {"a", "b"})
        self.assertTrue(
            per_hardware["rank_acceptability"].between(0.0, 1.0).all()
        )


if __name__ == "__main__":
    unittest.main()
