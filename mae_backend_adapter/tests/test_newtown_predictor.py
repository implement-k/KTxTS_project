"""신도시 wrapper의 빠른 계약 테스트."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from mae_backend_adapter import (
    DEFAULT_PERIOD,
    SUPPORTED_NEWTOWNS,
    SUPPORTED_PERIODS,
    InputValidationError,
    NewtownPredictor,
    NewtownPreprocessor,
)


class NewtownPredictorContractTests(unittest.TestCase):
    def test_city_only_defaults_to_final_period(self) -> None:
        preprocessor = NewtownPreprocessor("gyosan")
        self.assertEqual(preprocessor.newtown_name, "gyosan")
        self.assertEqual(preprocessor.period, DEFAULT_PERIOD)

    def test_unknown_city_and_period_fail(self) -> None:
        with self.assertRaises(InputValidationError):
            NewtownPreprocessor("unknown")
        with self.assertRaises(InputValidationError):
            NewtownPreprocessor("gyosan", "unknown")

    def test_all_supported_bundled_files_exist(self) -> None:
        root = Path(__file__).resolve().parents[1] / "newtown"
        for newtown in SUPPORTED_NEWTOWNS:
            with self.subTest(newtown=newtown):
                city_root = root / newtown
                self.assertTrue((city_root / "OD_dong_list_2023.xlsx").is_file())
                self.assertTrue((city_root / "dong_distance.csv").is_file())
                self.assertTrue((city_root / "dong_adjacency.pkl").is_file())
                self.assertTrue((city_root / "mask_code.json").is_file())
                for period in SUPPORTED_PERIODS:
                    self.assertTrue(
                        (city_root / f"static_features_{period}.csv").is_file()
                    )

    def test_wrapper_reuses_injected_predictor(self) -> None:
        predictor = mock.Mock()
        predictor.predict_from_tensors.return_value = {"od": []}
        wrapper = NewtownPredictor(predictor=predictor)
        prepared = mock.Mock()
        with mock.patch.object(NewtownPreprocessor, "prepare", return_value=prepared):
            result = wrapper.predict("gyosan")
        self.assertEqual(result, {"od": []})
        predictor.predict_from_tensors.assert_called_once()


if __name__ == "__main__":
    unittest.main()
