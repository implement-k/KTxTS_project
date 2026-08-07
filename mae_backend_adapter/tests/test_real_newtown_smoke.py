"""번들 데이터와 실제 checkpoint를 함께 실행하는 명시적 smoke test."""

from __future__ import annotations

import json
import os
import unittest

from mae_backend_adapter import (
    SUPPORTED_NEWTOWNS,
    SUPPORTED_PERIODS,
    NewtownPredictor,
)


@unittest.skipUnless(
    os.environ.get("MAE_RUN_REAL_NEWTOWN_SMOKE") == "1",
    "MAE_RUN_REAL_NEWTOWN_SMOKE=1일 때 실제 신도시 smoke를 실행합니다.",
)
class RealNewtownSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = NewtownPredictor(device="cpu")

    def test_real_newtown_smoke(self) -> None:
        for newtown in SUPPORTED_NEWTOWNS:
            for period in SUPPORTED_PERIODS:
                with self.subTest(newtown=newtown, period=period):
                    result = self.predictor.predict(newtown, period)
                    metadata = result["metadata"]
                    self.assertEqual(result["newtown"], newtown)
                    self.assertEqual(metadata["period"], period)
                    self.assertEqual(metadata["checkpoint"], "mae.pth")
                    self.assertGreater(metadata["node_count"], 0)
                    self.assertEqual(metadata["feature_count"], 20)
                    self.assertGreater(len(result["od"]), 0)
                    self.assertEqual(metadata["returned_od_count"], len(result["od"]))
                    json.dumps(result, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
