"""확정된 MAE v7 모델과 백엔드 공개 계약을 연결한다."""

from __future__ import annotations

import atexit
import json
import math
import subprocess
import sys
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn


AGE_RATIO_KEYS = frozenset({"0_19", "20_39", "40_64", "65_plus"})


class MAEAdapterError(RuntimeError):
    """어댑터가 처리할 수 있는 공통 오류."""


class InputValidationError(MAEAdapterError, ValueError):
    """백엔드 요청값이 공개 입력 계약을 만족하지 않을 때 발생한다."""


class TensorShapeError(InputValidationError):
    """전처리 결과나 모델 출력 tensor의 shape가 잘못됐을 때 발생한다."""


class CheckpointLoadError(MAEAdapterError):
    """체크포인트 또는 필수 보조 모델을 읽을 수 없을 때 발생한다."""


class CheckpointCompatibilityError(CheckpointLoadError):
    """체크포인트와 모델 구조가 strict 호환되지 않을 때 발생한다."""


class PreprocessingConfigurationError(MAEAdapterError):
    """공식 zone·배분·특성 설정이 없어 입력을 만들 수 없을 때 발생한다."""


@dataclass(frozen=True)
class ModelInputs:
    """외부 전처리기가 전달하는 batch 차원 없는 모델 입력."""

    x_static: Tensor
    x_od_masked: Tensor
    x_dist: Tensor
    mask: Tensor
    origin_codes: Collection[str]
    destination_codes: Collection[str]
    newtown_zone_codes: Collection[str]
    population_allocation_method: str
    output_transform: str = "log1p"
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PopulationPreprocessor(Protocol):
    """프로젝트의 공식 zone 및 feature 규칙을 구현하는 주입 계약."""

    supported_newtowns: Collection[str]

    def prepare(
        self,
        *,
        newtown: str,
        total_population: int,
        age_ratios: Mapping[str, float],
    ) -> ModelInputs:
        """사용자 입력을 학습 당시와 같은 tensor와 node 순서로 변환한다."""


ModelFactory = Callable[[Mapping[str, Tensor], Mapping[str, Any]], nn.Module]
SelfLoopPredictor = Callable[[Tensor], Sequence[float] | Tensor]


_LGBM_WORKER_CODE = r"""
import json
import sys
try:
    import lightgbm as lgb
    import numpy as np
    booster = lgb.Booster(model_file=sys.argv[1])
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        values = json.loads(line)
        prediction = booster.predict(np.asarray(values, dtype=np.float32))
        print(json.dumps(prediction.tolist(), allow_nan=False), flush=True)
except Exception as exc:
    print(json.dumps({"error": str(exc)}), flush=True)
    raise
"""


class _LightGBMWorker:
    """PyTorch와 OpenMP runtime이 충돌하지 않도록 LightGBM을 격리한다."""

    def __init__(self, model_path: Path) -> None:
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [sys.executable, "-c", _LGBM_WORKER_CODE, str(model_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready_line = self._process.stdout.readline() if self._process.stdout else ""
        try:
            ready = json.loads(ready_line)
        except (TypeError, json.JSONDecodeError):
            ready = {}
        if ready.get("ready") is not True:
            error = ready.get("error") or self._read_stderr()
            self.close()
            raise CheckpointLoadError(f"LightGBM worker 시작에 실패했습니다: {error}")
        atexit.register(self.close)

    def __call__(self, x_static: Tensor) -> Sequence[float]:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError(f"LightGBM worker가 종료됐습니다: {self._read_stderr()}")
            assert self._process.stdin is not None and self._process.stdout is not None
            self._process.stdin.write(json.dumps(x_static.tolist(), allow_nan=False) + "\n")
            self._process.stdin.flush()
            response = json.loads(self._process.stdout.readline())
        if isinstance(response, Mapping) and "error" in response:
            raise RuntimeError(str(response["error"]))
        return response

    def _read_stderr(self) -> str:
        if self._process.stderr is None:
            return "stderr 없음"
        if self._process.poll() is None:
            return "상세 오류 없음"
        return self._process.stderr.read().strip()

    def close(self) -> None:
        if self._process.poll() is None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()


class _TorchMAERunner:
    """모델 생성·strict load·tensor forward를 공개 API에서 격리한다."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        *,
        model_factory: ModelFactory | None,
        use_lgbm_self_loop: bool,
        lgbm_model_path: Path | None,
        self_loop_predictor: SelfLoopPredictor | None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        state_dict, model_config = self._load_checkpoint(model_path)
        factory = model_factory or self._build_v7_model
        try:
            model = factory(state_dict, model_config)
        except CheckpointCompatibilityError:
            raise
        except Exception as exc:
            raise CheckpointCompatibilityError(
                f"모델 구조 생성에 실패했습니다: {model_path.name}: {exc}"
            ) from exc
        if not isinstance(model, nn.Module):
            raise CheckpointCompatibilityError("model_factory는 torch.nn.Module을 반환해야 합니다.")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise CheckpointCompatibilityError(
                f"체크포인트가 모델 구조와 엄격 호환되지 않습니다: {model_path.name}: {exc}"
            ) from exc

        self.model = model.to(device).eval()
        self.num_nodes = self._infer_num_nodes(state_dict)
        feature_weight = state_dict.get("feature_embed.0.weight")
        self.num_features = (
            int(feature_weight.shape[1])
            if isinstance(feature_weight, Tensor) and feature_weight.ndim == 2
            else None
        )
        self.self_loop_predictor = self_loop_predictor
        self.lgbm_execution = "injected" if self_loop_predictor is not None else "disabled"
        self.lgbm_model_path = lgbm_model_path
        if use_lgbm_self_loop and self_loop_predictor is None:
            self.self_loop_predictor, self.lgbm_execution = self._load_lgbm(lgbm_model_path)

        # 이 모델의 head별 3차원 additive mask는 최신 PyTorch native MHA
        # fast path에서 NaN을 만들 수 있어 프로세스 단위로 안전한 경로를 사용한다.
        mha_backend = getattr(torch.backends, "mha", None)
        if mha_backend is not None and hasattr(mha_backend, "set_fastpath_enabled"):
            mha_backend.set_fastpath_enabled(False)

    def forward(self, tensors: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """eval 모델을 inference mode로 한 번 실행해 전체 N×N을 반환한다."""

        with torch.inference_mode():
            output = self.model(*tensors)
        pred_od = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(pred_od, Tensor):
            raise TensorShapeError("모델의 첫 번째 출력은 torch.Tensor여야 합니다.")
        return pred_od

    def replace_self_loops(self, pred_od: Tensor, x_static: Tensor) -> Tensor:
        """v7 설정처럼 LightGBM 예측으로 실수 단위 대각 성분을 덮는다."""

        if self.self_loop_predictor is None:
            return pred_od
        try:
            lgbm_input = x_static.detach().to(device="cpu", dtype=torch.float32)
            predicted = self.self_loop_predictor(lgbm_input)
            diagonal = torch.as_tensor(predicted, dtype=pred_od.dtype, device=pred_od.device)
        except Exception as exc:
            raise MAEAdapterError(f"LightGBM self-loop 예측에 실패했습니다: {exc}") from exc
        if tuple(diagonal.shape) != (pred_od.shape[0],):
            raise TensorShapeError(
                f"LightGBM self-loop 출력은 ({pred_od.shape[0]},)여야 하지만 "
                f"{tuple(diagonal.shape)}입니다."
            )
        if not torch.isfinite(diagonal).all():
            raise MAEAdapterError("LightGBM self-loop 출력에 NaN 또는 무한대가 있습니다.")
        result = pred_od.clone()
        index = torch.arange(pred_od.shape[0], device=pred_od.device)
        result[index, index] = torch.expm1(torch.clamp_min(diagonal, 0))
        return result

    @staticmethod
    def _load_checkpoint(path: Path) -> tuple[Mapping[str, Tensor], Mapping[str, Any]]:
        if not path.is_file():
            raise CheckpointLoadError(f"체크포인트 파일이 없습니다: {path}")
        try:
            with path.open("rb") as checkpoint_file:
                prefix = checkpoint_file.read(128)
        except OSError as exc:
            raise CheckpointLoadError(f"체크포인트를 읽을 수 없습니다: {path}: {exc}") from exc
        if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
            raise CheckpointLoadError(f"Git LFS pointer만 있고 실제 체크포인트가 없습니다: {path}")
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise CheckpointLoadError("weights_only 로딩을 지원하는 PyTorch가 필요합니다.") from exc
        except Exception as exc:
            raise CheckpointLoadError(f"체크포인트 로딩에 실패했습니다: {path}: {exc}") from exc

        model_config: Mapping[str, Any] = {}
        if isinstance(loaded, Mapping) and "state_dict" in loaded:
            state_dict = loaded["state_dict"]
            raw_config = loaded.get("model_config", {})
            if isinstance(raw_config, Mapping):
                model_config = raw_config
        else:
            state_dict = loaded
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise CheckpointLoadError("체크포인트에 비어 있지 않은 state_dict가 필요합니다.")
        if not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in state_dict.items()):
            raise CheckpointLoadError("state_dict는 문자열 key와 Tensor 값만 포함해야 합니다.")
        return state_dict, model_config

    @staticmethod
    def _infer_num_nodes(state_dict: Mapping[str, Tensor]) -> int | None:
        decoder = state_dict.get("decoder.2.weight")
        if not isinstance(decoder, Tensor) or decoder.ndim != 2 or decoder.shape[0] % 2:
            return None
        return int(decoder.shape[0] // 2)

    @classmethod
    def _build_v7_model(
        cls, state_dict: Mapping[str, Tensor], model_config: Mapping[str, Any]
    ) -> nn.Module:
        """state shape와 확정 argument를 조합해 feature/manage-version 모델을 만든다."""

        del model_config
        try:
            feature_weight = state_dict["feature_embed.0.weight"]
            num_nodes = cls._infer_num_nodes(state_dict)
            d_model, num_features = map(int, feature_weight.shape)
            nhead = int(state_dict["distance_bias.weight"].shape[1])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError("v7 모델 구조를 state shape에서 판별할 수 없습니다.") from exc
        if num_nodes is None:
            raise CheckpointCompatibilityError("decoder에서 고정 node 수를 판별할 수 없습니다.")
        layer_numbers = {
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith("transformer.layers.") and key.split(".")[2].isdigit()
        }
        od_embed_layers = 3 if "od_embed.4.weight" in state_dict else 2
        use_mask_channel = int(state_dict["od_embed.0.weight"].shape[1]) == num_nodes * 3
        use_self_loop_predictor = any(k.startswith("self_loop_predictor.") for k in state_dict)
        if (od_embed_layers, use_mask_channel, use_self_loop_predictor) != (3, True, True):
            raise CheckpointCompatibilityError("mae:v7 확정 구조(OD 3층/mask/self-loop)와 다릅니다.")
        try:
            from src.mae.models import SpatialODMAE
        except Exception as exc:
            raise CheckpointCompatibilityError(
                "src.mae.models.SpatialODMAE import에 실패했습니다. 저장소 루트가 Python "
                "import path에 있는지와 src/mae/models.py가 존재하는지 확인하세요."
            ) from exc
        return SpatialODMAE(
            num_nodes=num_nodes,
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=max(layer_numbers) + 1,
            od_embed_layers=3,
            use_distance_friction=False,
            use_self_loop_predictor=True,
            use_mask_channel=True,
        )

    @staticmethod
    def _load_lgbm(path: Path | None) -> tuple[SelfLoopPredictor, str]:
        if path is None or not path.is_file():
            raise CheckpointLoadError(f"v7에 필요한 LightGBM self-loop 파일이 없습니다: {path}")
        try:
            if sys.platform == "darwin":
                return _LightGBMWorker(path), "worker"

            import lightgbm as lgb

            booster = lgb.Booster(model_file=str(path))

            def predict(x_static: Tensor) -> Sequence[float]:
                return booster.predict(x_static.numpy())

            return predict, "in_process"
        except Exception as exc:
            if isinstance(exc, CheckpointLoadError):
                raise
            raise CheckpointLoadError(f"LightGBM 모델 로딩에 실패했습니다: {path}: {exc}") from exc


class MAEPredictor:
    """입력 검증·전처리 호출·신도시 OD 응답을 담당하는 공개 인터페이스."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        *,
        preprocessor: PopulationPreprocessor | None = None,
        supported_newtowns: Collection[str] | None = None,
        ratio_tolerance: float = 1e-6,
        model_factory: ModelFactory | None = None,
        use_lgbm_self_loop: bool = True,
        lgbm_model_path: str | Path | None = None,
        self_loop_predictor: SelfLoopPredictor | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = self._validate_device(device)
        self.ratio_tolerance = self._validate_ratio_tolerance(ratio_tolerance)
        self.preprocessor = preprocessor
        if supported_newtowns is None and preprocessor is not None:
            supported_newtowns = getattr(preprocessor, "supported_newtowns", None)
        self.supported_newtowns = frozenset(str(n) for n in (supported_newtowns or ()))
        lgbm_path = (
            Path(lgbm_model_path).expanduser().resolve()
            if lgbm_model_path is not None
            else self.model_path.with_name("best_lgbm_self_loop.txt")
        )
        self._runner = _TorchMAERunner(
            self.model_path,
            self.device,
            model_factory=model_factory,
            use_lgbm_self_loop=use_lgbm_self_loop,
            lgbm_model_path=lgbm_path,
            self_loop_predictor=self_loop_predictor,
        )
        # 기존 호출부와 진단 코드에서 읽을 수 있도록 read-only 성격으로 노출한다.
        self.model = self._runner.model
        self.num_nodes = self._runner.num_nodes
        self.num_features = self._runner.num_features

    def predict(
        self, *, newtown: str, total_population: int, age_ratios: Mapping[str, float]
    ) -> dict[str, Any]:
        """신도시 zone이 origin 또는 destination인 OD만 JSON 형태로 반환한다."""

        clean_newtown = self._validate_newtown(newtown)
        clean_population = self._validate_total_population(total_population)
        clean_ratios = self._validate_age_ratios(age_ratios)
        if self.preprocessor is None:
            raise PreprocessingConfigurationError(
                "신도시 zone, zone별 인구·연령·static feature, 전체 node 순서와 "
                "학습 scaler를 제공하는 프로젝트 전처리기가 필요합니다."
            )
        try:
            inputs = self.preprocessor.prepare(
                newtown=clean_newtown,
                total_population=clean_population,
                age_ratios=clean_ratios,
            )
        except (InputValidationError, PreprocessingConfigurationError):
            raise
        except Exception as exc:
            raise PreprocessingConfigurationError(f"운영 전처리에 실패했습니다: {exc}") from exc
        if not isinstance(inputs, ModelInputs):
            raise PreprocessingConfigurationError("preprocessor.prepare()는 ModelInputs를 반환해야 합니다.")
        return self.predict_from_tensors(
            inputs,
            request_metadata={
                "newtown": clean_newtown,
                "total_population": clean_population,
                "age_ratios": clean_ratios,
            },
        )

    def predict_from_tensors(
        self,
        inputs: ModelInputs,
        *,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """검증된 tensor를 실행하고 신도시 관련 OD만 추출한다."""

        matrix, metadata, origins, destinations, zones = self._run_full(
            inputs, request_metadata=request_metadata
        )
        zone_set = set(zones)
        od: list[dict[str, Any]] = []
        for origin_index, origin_code in enumerate(origins):
            origin_is_zone = origin_code in zone_set
            for destination_index, destination_code in enumerate(destinations):
                destination_is_zone = destination_code in zone_set
                if not (origin_is_zone or destination_is_zone):
                    continue
                movement_type = (
                    "internal"
                    if origin_is_zone and destination_is_zone
                    else "outflow" if origin_is_zone else "inflow"
                )
                od.append(
                    {
                        "origin_code": origin_code,
                        "destination_code": destination_code,
                        "predicted_trips": float(matrix[origin_index, destination_index]),
                        "movement_type": movement_type,
                    }
                )
        result = {
            "newtown": metadata.get("newtown"),
            "newtown_zone_codes": zones,
            "od": od,
            "metadata": metadata,
        }
        self._ensure_json(result)
        return result

    def predict_full_matrix_from_tensors(
        self,
        inputs: ModelInputs,
        *,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """진단용 저수준 API로 전체 N×N 행렬을 명시적으로 반환한다."""

        matrix, metadata, origins, destinations, _ = self._run_full(
            inputs, request_metadata=request_metadata
        )
        result = {
            "origin_codes": origins,
            "destination_codes": destinations,
            "od_matrix": matrix.tolist(),
            "metadata": metadata,
        }
        self._ensure_json(result)
        return result

    def _run_full(
        self,
        inputs: ModelInputs,
        *,
        request_metadata: Mapping[str, Any] | None,
    ) -> tuple[Tensor, dict[str, Any], list[str], list[str], list[str]]:
        tensors, origins, destinations, zones = self._prepare_tensors(inputs)
        pred_od = self._runner.forward(tensors)
        node_count = len(origins)
        if pred_od.ndim == 3 and pred_od.shape[0] == 1:
            pred_od = pred_od[0]
        if pred_od.ndim != 2 or tuple(pred_od.shape) != (node_count, node_count):
            raise TensorShapeError(
                f"모델 OD 출력은 ({node_count}, {node_count})여야 하지만 {tuple(pred_od.shape)}입니다."
            )
        if inputs.output_transform == "log1p":
            pred_od = torch.expm1(pred_od)
        elif inputs.output_transform != "identity":
            raise PreprocessingConfigurationError("output_transform은 'log1p' 또는 'identity'여야 합니다.")
        if not torch.isfinite(pred_od).all():
            raise MAEAdapterError("GNN 출력에 NaN 또는 무한대가 포함되어 있습니다.")
        pred_od = self._runner.replace_self_loops(pred_od, inputs.x_static)

        metadata: dict[str, Any] = dict(inputs.metadata)
        if request_metadata:
            metadata.update(request_metadata)
        metadata.update(
            {
                "model_version": self.model_path.stem,
                "checkpoint": self.model_path.name,
                "node_count": node_count,
                "device": str(self.device),
                "population_allocation_method": inputs.population_allocation_method,
                "output_transform": inputs.output_transform,
                "self_loop_policy": (
                    "lightgbm_override" if self._runner.self_loop_predictor else "gnn"
                ),
                "lightgbm_execution": self._runner.lgbm_execution,
                "negative_values_policy": "preserved_off_diagonal",
                "rounding_policy": "none",
            }
        )
        return pred_od.detach().to("cpu"), metadata, origins, destinations, zones

    def _prepare_tensors(
        self, inputs: ModelInputs
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], list[str], list[str], list[str]]:
        for name in ("x_static", "x_od_masked", "x_dist", "mask"):
            if not isinstance(getattr(inputs, name), Tensor):
                raise TensorShapeError(f"{name}는 torch.Tensor여야 합니다.")
        if inputs.x_static.ndim != 2:
            raise TensorShapeError(f"x_static은 (N, F)여야 하지만 {tuple(inputs.x_static.shape)}입니다.")
        node_count, feature_count = inputs.x_static.shape
        expected_square = (node_count, node_count)
        if tuple(inputs.x_od_masked.shape) != expected_square:
            raise TensorShapeError(f"x_od_masked는 {expected_square}여야 합니다.")
        if tuple(inputs.x_dist.shape) != expected_square:
            raise TensorShapeError(f"x_dist는 {expected_square}여야 합니다.")
        if tuple(inputs.mask.shape) != (node_count,):
            raise TensorShapeError(f"mask는 ({node_count},)여야 합니다.")
        if self.num_nodes is not None and node_count != self.num_nodes:
            raise TensorShapeError(
                f"mae:v7 체크포인트는 고정 {self.num_nodes} node지만 입력은 {node_count} node입니다."
            )
        if self.num_features is not None and feature_count != self.num_features:
            raise TensorShapeError(
                f"x_static feature 수는 {self.num_features}여야 하지만 {feature_count}입니다."
            )
        origins = [str(code) for code in inputs.origin_codes]
        destinations = [str(code) for code in inputs.destination_codes]
        zones = [str(code) for code in inputs.newtown_zone_codes]
        if len(origins) != node_count or len(destinations) != node_count:
            raise TensorShapeError("origin/destination code 수가 node 수와 다릅니다.")
        if len(set(origins)) != node_count or len(set(destinations)) != node_count:
            raise PreprocessingConfigurationError("node code는 각 축에서 중복될 수 없습니다.")
        if not zones or len(zones) != len(set(zones)):
            raise PreprocessingConfigurationError("중복 없는 newtown_zone_codes를 하나 이상 제공해야 합니다.")
        missing = set(zones) - (set(origins) & set(destinations))
        if missing:
            raise PreprocessingConfigurationError(f"전체 node 순서에 없는 신도시 zone입니다: {sorted(missing)}")
        if not inputs.population_allocation_method:
            raise PreprocessingConfigurationError("population_allocation_method를 명시해야 합니다.")
        tensors = (
            inputs.x_static.to(self.device, dtype=torch.float32).unsqueeze(0),
            inputs.x_od_masked.to(self.device, dtype=torch.float32).unsqueeze(0),
            inputs.x_dist.to(self.device, dtype=torch.float32).unsqueeze(0),
            inputs.mask.to(self.device, dtype=torch.bool).unsqueeze(0),
        )
        return tensors, origins, destinations, zones

    def _validate_newtown(self, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InputValidationError("newtown은 비어 있지 않은 문자열이어야 합니다.")
        clean = value.strip()
        if clean not in self.supported_newtowns:
            supported = ", ".join(sorted(self.supported_newtowns)) or "없음"
            raise InputValidationError(f"지원되지 않는 신도시입니다: {clean}. 현재 지원 목록: {supported}")
        return clean

    @staticmethod
    def _validate_total_population(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise InputValidationError("total_population은 0 이상의 정수여야 합니다.")
        return int(value)

    def _validate_age_ratios(self, value: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise InputValidationError("age_ratios는 mapping이어야 합니다.")
        keys = set(value)
        if keys != AGE_RATIO_KEYS:
            raise InputValidationError(
                f"age_ratios key가 정확하지 않습니다. 누락={sorted(AGE_RATIO_KEYS - keys)}, "
                f"알 수 없음={sorted(keys - AGE_RATIO_KEYS)}"
            )
        cleaned: dict[str, float] = {}
        for key in sorted(AGE_RATIO_KEYS):
            ratio = value[key]
            if isinstance(ratio, bool) or not isinstance(ratio, Real):
                raise InputValidationError(f"age_ratios[{key!r}]는 숫자여야 합니다.")
            number = float(ratio)
            if not math.isfinite(number) or number < 0:
                raise InputValidationError(f"age_ratios[{key!r}]는 유한한 0 이상의 숫자여야 합니다.")
            cleaned[key] = number
        if not math.isclose(sum(cleaned.values()), 1.0, rel_tol=0, abs_tol=self.ratio_tolerance):
            raise InputValidationError("age_ratios 합계는 1이어야 합니다.")
        return cleaned

    @staticmethod
    def _validate_device(value: str) -> torch.device:
        try:
            device = torch.device(value)
        except (TypeError, RuntimeError) as exc:
            raise InputValidationError(f"올바르지 않은 PyTorch device입니다: {value!r}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise InputValidationError("CUDA를 사용할 수 없습니다.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise InputValidationError("MPS를 사용할 수 없습니다.")
        if device.type not in {"cpu", "cuda", "mps"}:
            raise InputValidationError("device는 cpu, cuda 또는 mps여야 합니다.")
        return device

    @staticmethod
    def _validate_ratio_tolerance(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InputValidationError("ratio_tolerance는 0 이상의 유한한 숫자여야 합니다.")
        clean = float(value)
        if not math.isfinite(clean) or clean < 0:
            raise InputValidationError("ratio_tolerance는 0 이상의 유한한 숫자여야 합니다.")
        return clean

    @staticmethod
    def _ensure_json(result: Mapping[str, Any]) -> None:
        try:
            json.dumps(result, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise MAEAdapterError(f"반환값을 JSON으로 직렬화할 수 없습니다: {exc}") from exc
