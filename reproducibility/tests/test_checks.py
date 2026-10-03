from __future__ import annotations

import itertools
import math
import unittest

from src.compute import (
    all_sign_direct,
    all_sign_formula,
    error_coupling_checks,
    full_recall_coefficients,
    full_recall_uniform_channels,
    full_recall_uniform_exact,
    resurrection_coefficients,
    running_max_coefficients,
)


class FormulaChecks(unittest.TestCase):
    def test_all_sign_formula(self) -> None:
        values = (-2.0, -0.5, 0.5, 2.0)
        for a, h1, h2 in itertools.product(values, values, values):
            if a == 0.0:
                continue
            direct = all_sign_direct(h1, h2, a, 1.3, 0.7, 0.9)
            formula = all_sign_formula(h1, h2, a, 1.3, 0.7, 0.9)
            self.assertAlmostEqual(direct, formula, delta=2e-12)

    def test_resurrection(self) -> None:
        model_a, model_b = resurrection_coefficients(1.0, 1.0)
        self.assertEqual(model_a, 5.0 / 16.0)
        self.assertEqual(model_b, 7.0 / 16.0)

    def test_error_couplings(self) -> None:
        values = error_coupling_checks()
        self.assertEqual(values["aligned"], 2.5)
        self.assertEqual(values["anti_aligned"], 1.75)
        self.assertEqual(values["cross_moment_A"], values["cross_moment_B"])
        self.assertEqual(values["squared_min_A"], 23.0 / 4.0)
        self.assertEqual(values["squared_min_B"], 25.0 / 4.0)

    def test_running_maximum_coefficients(self) -> None:
        observed = running_max_coefficients()
        expected = (
            0.0209989023643558,
            0.0142824725470602,
            0.00788657453450204,
        )
        for value, target in zip(observed, expected):
            self.assertTrue(math.isclose(value, target, rel_tol=0.0, abs_tol=5e-12))

    def test_full_recall_search(self) -> None:
        date1, new_record, copied = full_recall_coefficients()
        self.assertEqual((date1, new_record, copied), (0.25, 0.125, 0.125))
        self.assertEqual(date1 + new_record + copied, 0.5)
        self.assertEqual(copied / (date1 + new_record), 1.0 / 3.0)
        for epsilon in (0.5, 0.2, 0.1, 0.01):
            channels = full_recall_uniform_channels(epsilon)
            self.assertEqual(channels[1], channels[2])
            exact = full_recall_uniform_exact(epsilon)
            expected = 0.5 * epsilon**2 - 0.25 * epsilon**4
            self.assertTrue(math.isclose(exact, expected, rel_tol=0.0, abs_tol=1e-18))


if __name__ == "__main__":
    unittest.main()
