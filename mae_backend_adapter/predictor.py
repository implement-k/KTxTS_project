"""최종 mae-year 모델과 Django 백엔드 공개 계약을 연결한다."""

from __future__ import annotations

import atexit
import importlib.util
import json
import math
import subprocess
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import torch
from torch import Tensor, nn


AGE_RATIO_KEYS = frozenset({"0_19", "20_59", "60_plus"})
UNMAPPED_NODE_CODE = "UNMAPPED"


class MAEAdapterError(RuntimeError):
    """어댑터가 처리하는 공통 오류."""


class InputValidationError(MAEAdapterError, ValueError):
    """백엔드 요청값이 공개 입력 계약을 만족하지 않을 때 발생한다."""


class TensorShapeError(InputValidationError):
    """모델 tensor의 shape, dtype 또는 값 범위가 잘못됐을 때 발생한다."""


class CheckpointLoadError(MAEAdapterError):
    """체크포인트 또는 필수 보조 모델을 읽을 수 없을 때 발생한다."""


class CheckpointCompatibilityError(CheckpointLoadError):
    """체크포인트와 최종 ODMAE 구조가 strict 호환되지 않을 때 발생한다."""


class PreprocessingConfigurationError(MAEAdapterError):
    """학습 당시와 같은 전처리 입력을 구성할 수 없을 때 발생한다."""


@dataclass(frozen=True)
class ModelInputs:
    """백엔드 ``PopulationPreprocessor``가 만드는 batch 없는 입력 DTO.

    ``N``은 외부 요청값이 아니라 ``x_static.shape[0]``에서 정해진다. 원본 파일
    순서는 달라도 되지만, DTO를 만들 때는 canonical node code 목록으로 static
    feature, OD·거리·인접 행렬, mask와 code를 모두 reindex해야 한다.

    - ``x_static``: ``(N, F)`` floating tensor. Dataset과 같은 feature 이름 정렬과
      학습 scaler 결과가 적용된 값을 백엔드 전처리기가 제공한다.
    - ``x_od_masked``, ``x_dist``, ``a_spatial``: 각각 ``(N, N)`` floating tensor.
      OD·거리 전처리와 인접 행렬 구성이 끝난 값이며 백엔드 전처리기가 제공한다.
    - ``mask``, ``active_node_mask``: 각각 ``(N,)`` bool tensor. 현재 운영은 선택
      신도시만 ``mask=True``이고 모두 active다. 일반 계약은 비활성 node도 지원한다.
    - ``origin_codes``, ``destination_codes``: 길이 ``N``의 code sequence. 수치
      전처리 대상이 아니며 canonical node 순서로 제공한다.
    - ``newtown_zone_codes``: 신도시 code collection. 백엔드 도시 설정이 제공한다.
    - ``population_allocation_method``: 배분 설정 이름 문자열. 백엔드 설정이 제공한다.
    - ``output_transform``: 모델 출력 변환 이름 문자열. AI 팀 모델 계약이 제공한다.
    - ``metadata``: 응답에 전달할 mapping. 연도 등 백엔드 실행 문맥이 제공한다.

    Adapter는 shape과 code 길이를 검증할 뿐 의미적 순서는 알 수 없다. reindex는
    concrete preprocessor의 책임이다. 필수 데이터·설정이 없으면 합성값이나
    프론트 더미 값을 쓰지 말고 ``PreprocessingConfigurationError``를 발생시켜야 한다.
    """

    x_static: Tensor
    x_od_masked: Tensor
    x_dist: Tensor
    a_spatial: Tensor
    mask: Tensor
    active_node_mask: Tensor
    origin_codes: Sequence[Any]
    destination_codes: Sequence[Any]
    newtown_zone_codes: Collection[Any]
    population_allocation_method: str
    output_transform: str = "log1p"
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PopulationPreprocessor(Protocol):
    """백엔드 요청을 도시별 전처리 완료 ``ModelInputs``로 바꾸는 계약.

    ``newtown``에 맞는 도시별 데이터를 선택하고 canonical node code로 static,
    OD·거리·인접 행렬, mask와 code를 재정렬한다. static feature는
    ``dong_code``, ``dong_name``을 제외한 column 이름을 Dataset과 같이 정렬한다.

    Scaler는 요청 데이터로 fit하지 않고 2023 배포 artifact를 process 당 한 번
    로드해 재사용한다. 18개 raw feature를 artifact schema로 정렬·변환한 뒤
    ``is_masked``, ``is_merged``를 붙여야 한다.
    현재 운영은 모든 node를 active로 두고 선택 신도시만 mask하며, mask된
    OD 행·열을 0으로 만든다.

    단계별 인구, 기본 연령 비율, static feature 기본값과 가상 node별
    배분값은 미정이다. Adapter에 하드코딩하거나 프론트 더미 값으로 추론하지
    않는다. 필수 데이터·설정이 없으면 ``PreprocessingConfigurationError``를 발생시킨다.
    """

    supported_newtowns: Collection[str]

    def prepare(
        self,
        *,
        newtown: str,
        total_population: int,
        age_ratios: Mapping[str, float],
    ) -> ModelInputs:
        """명시적 인구·연령 입력으로 canonical 순서 DTO를 만들고, 필수 데이터가 없으면 실패한다."""


ModelFactory = Callable[[Mapping[str, Tensor], Mapping[str, Any]], nn.Module]
SelfLoopPredictor = Callable[[Tensor], Sequence[float] | Tensor]


def _normalize_node_code(value: Any) -> str:
    if value is None:
        return UNMAPPED_NODE_CODE
    clean = str(value).strip()
    return clean or UNMAPPED_NODE_CODE


class ODOutputAdapter:
    """모델 행렬을 백엔드 OD 행 목록으로 변환한다."""

    def adapt(
        self,
        matrix: Tensor,
        *,
        origin_codes: Sequence[str],
        destination_codes: Sequence[str],
        newtown_zone_codes: Sequence[str],
        active_node_mask: Sequence[bool],
    ) -> list[dict[str, Any]]:
        node_count = len(origin_codes)
        if matrix.ndim != 2 or tuple(matrix.shape) != (node_count, node_count):
            raise TensorShapeError(
                f"출력 Adapter 입력은 ({node_count}, {node_count})여야 하지만 "
                f"{tuple(matrix.shape)}입니다."
            )
        if len(destination_codes) != node_count or len(active_node_mask) != node_count:
            raise TensorShapeError("출력 code 또는 active_node_mask 수가 node 수와 다릅니다.")
        if not torch.isfinite(matrix).all():
            raise MAEAdapterError("출력 Adapter 입력에 NaN 또는 무한대가 있습니다.")

        zone_set = set(newtown_zone_codes)
        totals: OrderedDict[tuple[str, str, str], float] = OrderedDict()
        for origin_index, origin_code in enumerate(origin_codes):
            if not active_node_mask[origin_index]:
                continue
            origin_is_zone = origin_code in zone_set
            for destination_index, destination_code in enumerate(destination_codes):
                if not active_node_mask[destination_index]:
                    continue
                destination_is_zone = destination_code in zone_set
                if not (origin_is_zone or destination_is_zone):
                    # 외부 -> 외부는 신도시 대시보드 계산 대상이 아니다.
                    continue
                movement_type = (
                    "internal"
                    if origin_is_zone and destination_is_zone
                    else "outflow" if origin_is_zone else "inflow"
                )
                key = (origin_code, destination_code, movement_type)
                # 음수는 모델 후처리에서 이미 0으로 보정하지만 Adapter 단독 사용도 안전하게 한다.
                value = max(float(matrix[origin_index, destination_index]), 0.0)
                totals[key] = totals.get(key, 0.0) + value

        # node code 매핑 결과가 중복되면 먼저 합산하고, 최종 0 OD는 서비스 결과에서 뺀다.
        return [
            {
                "origin_code": origin,
                "destination_code": destination,
                "predicted_trips": trips,
                "movement_type": movement,
            }
            for (origin, destination, movement), trips in totals.items()
            if trips > 0.0
        ]


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
    """macOS에서 PyTorch와 LightGBM의 OpenMP runtime을 별도 process로 격리한다."""

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
    """최종 모델 생성, strict load 및 6-tensor forward를 격리한다."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        *,
        model_factory: ModelFactory | None,
        model_module_path: Path,
        use_lgbm_self_loop: bool,
        lgbm_model_path: Path | None,
        self_loop_predictor: SelfLoopPredictor | None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self._inference_lock = threading.RLock()
        state_dict, model_config = self._load_checkpoint(model_path)
        factory = model_factory or (
            lambda state, config: self._build_mae_year_model(
                state, config, model_module_path=model_module_path
            )
        )
        try:
            model = factory(state_dict, model_config)
        except CheckpointCompatibilityError:
            raise
        except Exception as exc:
            raise CheckpointCompatibilityError(
                f"최종 ODMAE 구조 생성에 실패했습니다: {model_path.name}: {exc}"
            ) from exc
        if not isinstance(model, nn.Module):
            raise CheckpointCompatibilityError("model_factory는 torch.nn.Module을 반환해야 합니다.")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise CheckpointCompatibilityError(
                f"체크포인트가 최종 모델 구조와 strict 호환되지 않습니다: "
                f"{model_path.name}: {exc}"
            ) from exc

        self.model = model.to(device).eval()
        feature_weight = state_dict.get("feature_embed.0.weight")
        self.num_features = (
            int(feature_weight.shape[1])
            if isinstance(feature_weight, Tensor) and feature_weight.ndim == 2
            else None
        )
        # 이전 호출부 호환용 속성이다. N은 checkpoint가 아니라 매 ModelInputs에서 결정된다.
        self.num_nodes: int | None = None
        self.self_loop_predictor = self_loop_predictor
        self.lgbm_execution = "injected" if self_loop_predictor is not None else "disabled"
        self.lgbm_model_path = lgbm_model_path
        if use_lgbm_self_loop and self_loop_predictor is None:
            self.self_loop_predictor, self.lgbm_execution = self._load_lgbm(lgbm_model_path)

        # head별 3차원 additive mask가 native MHA fast path에서 NaN이 되는 것을 막는다.
        mha_backend = getattr(torch.backends, "mha", None)
        if mha_backend is not None and hasattr(mha_backend, "set_fastpath_enabled"):
            mha_backend.set_fastpath_enabled(False)

    def forward(self, tensors: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """eval 모델을 inference mode로 한 번 실행해 batch 포함 전체 N×N을 반환한다."""

        with self._inference_lock, torch.inference_mode():
            output = self.model(*tensors)
        pred_od = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(pred_od, Tensor):
            raise TensorShapeError("모델의 첫 번째 출력은 torch.Tensor여야 합니다.")
        return pred_od

    def replace_self_loops(self, pred_od: Tensor, x_static: Tensor) -> Tensor:
        """최종 평가 설정처럼 LightGBM의 log 출력으로 대각 성분을 덮는다."""

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
        state_dict: Any = loaded
        if isinstance(loaded, Mapping):
            if isinstance(loaded.get("model_state_dict"), Mapping):
                state_dict = loaded["model_state_dict"]
            elif isinstance(loaded.get("state_dict"), Mapping):
                state_dict = loaded["state_dict"]
            raw_config = loaded.get("model_config", {})
            if isinstance(raw_config, Mapping):
                model_config = raw_config
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise CheckpointLoadError("체크포인트에 비어 있지 않은 state_dict가 필요합니다.")
        if not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in state_dict.items()):
            raise CheckpointLoadError("state_dict는 문자열 key와 Tensor 값만 포함해야 합니다.")
        return state_dict, model_config

    @classmethod
    def _build_mae_year_model(
        cls,
        state_dict: Mapping[str, Tensor],
        model_config: Mapping[str, Any],
        *,
        model_module_path: Path,
    ) -> nn.Module:
        """state shape와 최종 학습 설정으로 src/mae-year ODMAE를 만든다."""

        try:
            feature_weight = state_dict["feature_embed.0.weight"]
            d_model, num_features = map(int, feature_weight.shape)
            nhead = int(state_dict["distance_bias.weight"].shape[1])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                "최종 ODMAE 구조를 checkpoint state shape에서 판별할 수 없습니다."
            ) from exc
        layer_numbers = {
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith("transformer.layers.") and key.split(".")[2].isdigit()
        }
        if not layer_numbers:
            raise CheckpointCompatibilityError("transformer layer 수를 판별할 수 없습니다.")
        use_self_loop_predictor = any(
            key.startswith("self_loop_predictor.") for key in state_dict
        )
        use_distance_friction = bool(model_config.get("use_distance_friction", False))
        module = cls._load_model_module(model_module_path)
        model_class = getattr(module, "ODMAE", None)
        if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
            raise CheckpointCompatibilityError(
                f"최종 모델 파일에 torch.nn.Module ODMAE가 없습니다: {model_module_path}"
            )
        return model_class(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=max(layer_numbers) + 1,
            use_distance_friction=use_distance_friction,
            use_self_loop_predictor=use_self_loop_predictor,
        )

    @staticmethod
    def _load_model_module(path: Path) -> ModuleType:
        if not path.is_file():
            raise CheckpointCompatibilityError(
                f"최종 모델 파일이 없습니다: {path}. src/mae-year/models.py를 함께 배포하세요."
            )
        module_name = f"_mae_year_models_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise CheckpointCompatibilityError(f"최종 모델 module spec을 만들 수 없습니다: {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise CheckpointCompatibilityError(
                f"최종 모델 module import에 실패했습니다: {path}: {exc}"
            ) from exc
        return module

    @staticmethod
    def _load_lgbm(path: Path | None) -> tuple[SelfLoopPredictor, str]:
        if path is None or not path.is_file():
            raise CheckpointLoadError(f"LightGBM self-loop 파일이 없습니다: {path}")
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
    """입력 검증, 최종 모델 실행 및 백엔드 OD 변환을 담당하는 Provider."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        *,
        preprocessor: PopulationPreprocessor | None = None,
        supported_newtowns: Collection[str] | None = None,
        ratio_tolerance: float = 1e-6,
        model_factory: ModelFactory | None = None,
        model_module_path: str | Path | None = None,
        use_lgbm_self_loop: bool = True,
        lgbm_model_path: str | Path | None = None,
        self_loop_predictor: SelfLoopPredictor | None = None,
        output_adapter: ODOutputAdapter | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = self._validate_device(device)
        self.ratio_tolerance = self._validate_ratio_tolerance(ratio_tolerance)
        self.preprocessor = preprocessor
        if supported_newtowns is None and preprocessor is not None:
            supported_newtowns = getattr(preprocessor, "supported_newtowns", None)
        self.supported_newtowns = frozenset(str(name) for name in (supported_newtowns or ()))
        repository_root = Path(__file__).resolve().parents[1]
        module_path = (
            Path(model_module_path).expanduser().resolve()
            if model_module_path is not None
            else repository_root / "src" / "mae-year" / "models.py"
        )
        lgbm_path = (
            Path(lgbm_model_path).expanduser().resolve()
            if lgbm_model_path is not None
            else self.model_path.with_name("best_lgbm_self_loop.txt")
        )
        self._runner = _TorchMAERunner(
            self.model_path,
            self.device,
            model_factory=model_factory,
            model_module_path=module_path,
            use_lgbm_self_loop=use_lgbm_self_loop,
            lgbm_model_path=lgbm_path,
            self_loop_predictor=self_loop_predictor,
        )
        self.output_adapter = output_adapter or ODOutputAdapter()
        self.model = self._runner.model
        self.num_nodes = self._runner.num_nodes
        self.num_features = self._runner.num_features

    def predict(
        self, *, newtown: str, total_population: int, age_ratios: Mapping[str, float]
    ) -> dict[str, Any]:
        """기존 백엔드 공개 요청 계약을 유지해 신도시 관련 OD를 반환한다."""

        clean_newtown = self._validate_newtown(newtown)
        clean_population = self._validate_total_population(total_population)
        clean_ratios = self._validate_age_ratios(age_ratios)
        if self.preprocessor is None:
            raise PreprocessingConfigurationError(
                "신도시 zone, 20개 static feature, 학습 scaler, OD/거리/인접 행렬과 "
                "node 순서를 제공하는 프로젝트 전처리기가 필요합니다."
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
        """전처리 완료 DTO를 실행하는 Provider 내부/테스트용 진입점."""

        matrix, metadata, origins, destinations, zones, active = self._run_full(
            inputs, request_metadata=request_metadata
        )
        od = self.output_adapter.adapt(
            matrix,
            origin_codes=origins,
            destination_codes=destinations,
            newtown_zone_codes=zones,
            active_node_mask=active,
        )
        metadata["returned_od_count"] = len(od)
        result = {
            "newtown": metadata.get("newtown"),
            "newtown_zone_codes": zones,
            "od": od,
            "metadata": metadata,
        }
        self._ensure_json(result)
        return result

    def _run_full(
        self,
        inputs: ModelInputs,
        *,
        request_metadata: Mapping[str, Any] | None,
    ) -> tuple[Tensor, dict[str, Any], list[str], list[str], list[str], list[bool]]:
        tensors, origins, destinations, zones, active = self._prepare_tensors(inputs)
        pred_od = self._runner.forward(tensors)
        # _prepare_tensors에서 검증한 것과 같은 x_static 기반 동적 N을 사용한다.
        node_count = inputs.x_static.shape[0]
        if pred_od.ndim == 3 and pred_od.shape[0] == 1:
            pred_od = pred_od[0]
        if pred_od.ndim != 2 or tuple(pred_od.shape) != (node_count, node_count):
            raise TensorShapeError(
                f"모델 OD 출력은 ({node_count}, {node_count})여야 하지만 "
                f"{tuple(pred_od.shape)}입니다."
            )
        if inputs.output_transform == "log1p":
            pred_od = torch.expm1(pred_od)
        elif inputs.output_transform != "identity":
            raise PreprocessingConfigurationError("output_transform은 'log1p' 또는 'identity'여야 합니다.")
        if not torch.isfinite(pred_od).all():
            raise MAEAdapterError("MAE 출력 변환 결과에 NaN 또는 무한대가 있습니다.")
        pred_od = torch.clamp_min(pred_od, 0)
        pred_od = self._runner.replace_self_loops(pred_od, inputs.x_static)

        metadata: dict[str, Any] = dict(inputs.metadata)
        if request_metadata:
            metadata.update(request_metadata)
        metadata.update(
            {
                "model_family": "mae-year/ODMAE",
                "model_version": self.model_path.stem,
                "checkpoint": self.model_path.name,
                "node_count": node_count,
                "feature_count": inputs.x_static.shape[1],
                "device": str(self.device),
                "population_allocation_method": inputs.population_allocation_method,
                "output_transform": inputs.output_transform,
                "self_loop_policy": (
                    "lightgbm_override" if self._runner.self_loop_predictor else "mae"
                ),
                "lightgbm_execution": self._runner.lgbm_execution,
                "negative_values_policy": "clamped_to_zero",
                "duplicate_od_policy": "summed",
                "zero_od_policy": "excluded",
                "external_to_external_policy": "excluded",
                "unmapped_node_code": UNMAPPED_NODE_CODE,
                "rounding_policy": "none",
            }
        )
        return (
            pred_od.detach().to("cpu"),
            metadata,
            origins,
            destinations,
            zones,
            active,
        )

    def _prepare_tensors(
        self, inputs: ModelInputs
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        list[str],
        list[str],
        list[str],
        list[bool],
    ]:
        float_names = ("x_static", "x_od_masked", "x_dist", "a_spatial")
        for name in (*float_names, "mask", "active_node_mask"):
            if not isinstance(getattr(inputs, name), Tensor):
                raise TensorShapeError(f"{name}는 torch.Tensor여야 합니다.")
        for name in float_names:
            tensor = getattr(inputs, name)
            if not torch.is_floating_point(tensor):
                raise TensorShapeError(f"{name}는 floating-point tensor여야 합니다.")
            if not torch.isfinite(tensor).all():
                raise TensorShapeError(f"{name}에 NaN 또는 무한대가 있습니다.")
        if inputs.mask.dtype != torch.bool or inputs.active_node_mask.dtype != torch.bool:
            raise TensorShapeError("mask와 active_node_mask dtype은 torch.bool이어야 합니다.")
        if inputs.x_static.ndim != 2:
            raise TensorShapeError(f"x_static은 (N, F)여야 하지만 {tuple(inputs.x_static.shape)}입니다.")
        # 동적 N의 유일한 기준. N 자체를 API나 checkpoint 설정에서 받지 않는다.
        node_count, feature_count = inputs.x_static.shape
        expected_square = (node_count, node_count)
        for name in ("x_od_masked", "x_dist", "a_spatial"):
            if tuple(getattr(inputs, name).shape) != expected_square:
                raise TensorShapeError(f"{name}는 {expected_square}여야 합니다.")
        if tuple(inputs.mask.shape) != (node_count,):
            raise TensorShapeError(f"mask는 ({node_count},)여야 합니다.")
        if tuple(inputs.active_node_mask.shape) != (node_count,):
            raise TensorShapeError(f"active_node_mask는 ({node_count},)여야 합니다.")
        if self.num_features is not None and feature_count != self.num_features:
            raise TensorShapeError(
                f"x_static feature 수는 {self.num_features}여야 하지만 {feature_count}입니다."
            )
        if torch.any(inputs.x_od_masked < 0):
            raise TensorShapeError("x_od_masked는 log1p 공간의 0 이상 값이어야 합니다.")
        if torch.any(inputs.x_dist < 0):
            raise TensorShapeError("x_dist는 log1p 공간의 0 이상 값이어야 합니다.")
        if torch.any((inputs.a_spatial < 0) | (inputs.a_spatial > 1)):
            raise TensorShapeError("a_spatial 값은 0~1 범위여야 합니다.")

        origins = [_normalize_node_code(code) for code in inputs.origin_codes]
        destinations = [_normalize_node_code(code) for code in inputs.destination_codes]
        zones = [_normalize_node_code(code) for code in inputs.newtown_zone_codes]
        if len(origins) != node_count or len(destinations) != node_count:
            raise TensorShapeError("origin/destination code 수가 node 수와 다릅니다.")
        if origins != destinations:
            raise PreprocessingConfigurationError(
                "origin_codes와 destination_codes는 동일한 node 순서여야 합니다."
            )
        if not zones or len(zones) != len(set(zones)):
            raise PreprocessingConfigurationError(
                "중복 없는 newtown_zone_codes를 하나 이상 제공해야 합니다."
            )
        if UNMAPPED_NODE_CODE in zones:
            raise PreprocessingConfigurationError("신도시 zone 자체는 UNMAPPED일 수 없습니다.")
        missing = set(zones) - set(origins)
        if missing:
            raise PreprocessingConfigurationError(
                f"전체 node code에 없는 신도시 zone입니다: {sorted(missing)}"
            )
        zone_indices = [index for index, code in enumerate(origins) if code in set(zones)]
        if not all(bool(inputs.mask[index]) for index in zone_indices):
            raise PreprocessingConfigurationError("모든 신도시 zone node는 mask=True여야 합니다.")
        if not all(bool(inputs.active_node_mask[index]) for index in zone_indices):
            raise PreprocessingConfigurationError("신도시 zone node는 active 상태여야 합니다.")
        inactive = ~inputs.active_node_mask
        if torch.any(inputs.x_od_masked[inactive, :] != 0) or torch.any(
            inputs.x_od_masked[:, inactive] != 0
        ):
            raise PreprocessingConfigurationError(
                "비활성 node의 x_od_masked 행과 열은 0이어야 합니다."
            )
        if torch.any(inputs.a_spatial[inactive, :] != 0) or torch.any(
            inputs.a_spatial[:, inactive] != 0
        ):
            raise PreprocessingConfigurationError("비활성 node의 a_spatial 행과 열은 0이어야 합니다.")
        if not inputs.population_allocation_method:
            raise PreprocessingConfigurationError("population_allocation_method를 명시해야 합니다.")

        active = [bool(value) for value in inputs.active_node_mask.tolist()]
        tensors = (
            inputs.x_static.to(self.device, dtype=torch.float32).unsqueeze(0),
            inputs.x_od_masked.to(self.device, dtype=torch.float32).unsqueeze(0),
            inputs.x_dist.to(self.device, dtype=torch.float32).unsqueeze(0),
            inputs.a_spatial.to(self.device, dtype=torch.float32).unsqueeze(0),
            inputs.mask.to(self.device, dtype=torch.bool).unsqueeze(0),
            inputs.active_node_mask.to(self.device, dtype=torch.bool).unsqueeze(0),
        )
        return tensors, origins, destinations, zones, active

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


class MAEProvider(MAEPredictor):
    """Django의 Mock Provider 교체 지점에서 사용할 명시적 이름."""
