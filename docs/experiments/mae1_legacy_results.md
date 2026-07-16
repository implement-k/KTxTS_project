# MAE1 Loss 탐색 — 과거 실험 기록 정리 (Notion + Git 재구성)

이 문서는 `experiment/loss-comparison` 작업을 시작하기 전, Codex 코드 리뷰 결과와
Notion에 남아있던 과거 실험 기록을 대조하여 정리한 것이다. 새 loss 후보를 만들기 전에
"실제로 무엇과 비교해야 하는가"를 고정하기 위한 기준 문서다.

## 1. 확인된 과거 최고 성능 (MAE1 단독, loss 실험 대상)

| 항목 | 값 |
|---|---|
| CPC | 0.5434 |
| RMSE | 924.49 |
| epochs | 50 |
| batch_size | 32 |
| min_mask_size / max_mask_size | 3 / 10 |
| learning_rate (Notion 기록) | 1e-3 |
| alpha schedule | heavy-tail, 10.0 → 1.0로 선형 감소 |
| optimizer | AdamW |
| scheduler | OneCycleLR |
| 특이사항 | log1p → exp → log1p 변환 버그 수정 이후 결과 |
| 실측 소요시간 | Colab CUDA 기준 epoch당 약 54–56초 |

이 결과가 이번 실험의 **유일한 loss 비교 기준선(baseline)** 이다.

## 2. 과거 기록 중 이번 loss 비교에서 제외되는 결과

아래 세 결과는 loss 함수 자체의 성능이 아니라 **앙상블/다른 모델 구조**의 결과이므로,
이번 "MAE1 단독 loss 개선" 실험의 순위 산정에는 사용하지 않는다. 참고용으로만 남긴다.

| 구성 | CPC | RMSE |
|---|---|---|
| MAE1 + LGBM 앙상블 | 0.6443 | 779.53 |
| Dynamic Soft-Threshold 앙상블 | 0.6819 | 697.38 |
| LGBM 단독 | 0.6432 | 698.92 |

## 3. Git 이력에서 복원한 `dynamic_weighted_mse` 정확한 수식

Notion 기록의 "heavy-tail alpha 10.0→1.0 가중치"는 현재 코드베이스의
`WeightedMSELossWrapper`(고정 alpha=1.5)와는 다른 별개의 구현이었다. 이를 그대로 재현하기 위해
`git log --oneline --all -- src/model/loss.py` 로 이력을 추적한 뒤, `git show <commit>:<path>`로
commit `a17b361` 시점의 파일 내용을 직접 확인하여 아래 클래스로 복원했다 (`src/model/loss.py`의
`DynamicWeightedMSELoss`):

```python
class DynamicWeightedMSELoss(nn.Module):
    def __init__(self, alpha=10.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred_log, target_log, mask=None):
        if mask is not None:
            pred_log = pred_log[mask]
            target_log = target_log[mask]
        real_target = torch.expm1(target_log)
        weight = 1.0 + self.alpha * real_target
        weight = weight / (weight.mean() + self.eps)
        loss = ((pred_log - target_log) ** 2) * weight
        return loss.mean()
```

`self.alpha`는 `train.py`가 매 epoch마다 외부에서 직접 대입한다:

```python
if args.loss == 'dynamic_weighted_mse':
    criterion.alpha = max(1.0, 10.0 * (1.0 - progress))  # progress = epoch / (epochs-1)
```

즉 alpha는 학습 진행률에 따라 10.0에서 1.0까지 선형으로 감소한다. 이 부분은 git 이력에
그대로 남아있어 **정확히 복원 가능했다**.

## 4. Git 이력으로 복원되지 않은 설정 (주의해서 사용할 것)

- **learning_rate**: Notion 기록은 `1e-3`이지만, 같은 시점 git 커밋의 실제 하드코딩된 값은
  `1e-4`였다. `TRAIN_CONFIG['learning_rate']` 키 자체는 `1e-3`으로 설정되어 있었으나 당시 학습
  루프가 이 키를 참조하지 않고 별도로 `1e-4`를 하드코딩해서 사용했을 가능성이 있다 — 즉 Notion
  기록이 "설정 파일에 적힌 값"을 옮겨 적었을 뿐 실제 실행값이 아니었을 가능성이 있다. 이 불일치는
  해소되지 않았으므로, Stage 0 재현 시 실제 실행 커밋의 값(`1e-4`)을 우선 사용하고 결과를 CPC
  0.5434와 비교할 때 이 차이를 원인 후보로 함께 기록한다.
- **정확한 random seed**: 당시 실행에 사용된 seed 값은 로그/커밋 어디에도 남아있지 않아 특정할
  수 없다. 이번 재현에서는 `seed=42`를 사용하고, 완전히 동일한 seed가 아니었을 수 있음을 명시한다.
- **체크포인트 자체의 메타데이터**: 과거 `.pth` 체크포인트를 `zipfile` + `pickletools.dis`로
  안전하게(역직렬화하지 않고) 구조만 확인했으나, 학습 조건을 담은 메타데이터는 포함되어 있지 않았다.
- **데이터 분할(정확한 train/test 노드 목록)이 현재와 동일했는지**: 당시 실행이 정확히 어떤
  노드 집합을 test로 사용했는지 로그로 확인할 수 없다. 현재 `ODDataset`의 `test_indices`
  (동탄/위례/검단)와 동일하다고 가정하고 진행한다.

## 5. 이번 실험에서의 사용 방식

- `dynamic_weighted_mse` = 위 수식대로 정확히 복원한 **유일한 baseline**.
- `weighted_mse_fixed` (alpha=1.5 고정)는 기존에 구현되어 있던 별개의 손실이며, baseline이
  아니라 **참고용(reference)** 으로만 함께 기록한다.
- Stage 0(legacy protocol, 50 epoch, seed=42)에서 `dynamic_weighted_mse`로 CPC 0.5434 /
  RMSE 924.49 재현을 시도하고, 정확히 일치하지 않더라도 차이와 추정 원인(위 4번 항목)을
  결과에 함께 기록한다.
