
from model import DeepGravityFFN
import torch, time, os, sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np

SRC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = os.path.dirname(SRC_PATH)
EVAL_PATH = os.path.join(SRC_PATH, 'evaluation')
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, EVAL_PATH)

from fixed_eval_utils import apply_merge_events # type: ignore

def build_od_features(X_static: np.ndarray, X_dist: np.ndarray) -> np.ndarray:
    """
    모든 (i,j) 쌍에 대해 [feat_i | feat_j | dist_ij] 를 구성.
    반환: (N*N, 2F+1) float32
    """
    N, _ = X_static.shape
    feat_o = np.repeat(X_static, N, axis=0)
    feat_d = np.tile(X_static, (N, 1))
    dist_flat = X_dist.flatten()[:, None]   
    return np.concatenate([feat_o, feat_d, dist_flat], axis=1).astype(np.float32)


def cpc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = 2 * np.sum(np.minimum(y_true, y_pred))
    den = np.sum(y_true) + np.sum(y_pred)
    return float(num / den) if den > 0 else 0.0

def train(model: DeepGravityFFN,
        optimizer: torch.optim.Optimizer,
        data_list: list,
        device: torch.device,
        n_epochs: int = 15,
        batch_size: int = 32):
    """
    data_list[i] = {
        'X_static': (N, F)   float32  normalized
        'X_dist':   (N, N)   float32
        'X_OD':     (N, N)   float32  ground-truth
        'train_idx': array of int
    }

    배치 처리 방식:
        B개 origin을 묶어서 한 번에 forward/backward.
        - feat: (B, N, 2F+1) -> reshape (B*N, 2F+1) -> FFN -> (B*N,) -> reshape (B, N)
        - loss:  각 행에 multinomial_nll -> sum over B
    """
    # 전체 데이터를 미리 텐서로 변환 (epoch마다 반복 변환 방지)
    ds_tensors = []
    for ds in data_list:
        X_s = torch.tensor(ds['X_static'], dtype=torch.float32, device=device)  # (N, F)
        X_d = torch.tensor(ds['X_dist'], dtype=torch.float32, device=device)    # (N, N) raw distance
        X_od = torch.tensor(ds['X_OD'], dtype=torch.float32, device=device)     # (N, N)
        train_idx = torch.tensor(ds['train_idx'], dtype=torch.long, device=device)
        ds_tensors.append((X_s, X_d, X_od, train_idx))

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_origins  = 0
        t0 = time.time()

        for (X_s, X_d, X_od, train_idx) in ds_tensors:
            N, F = X_s.shape
            # 발생량 없는 origin 필터링
            O_vals = X_od[train_idx].sum(dim=1)   # (n_train,)
            valid  = train_idx[O_vals >= 1]        # 발생량 >= 1인 origin만

            # epoch마다 순서 셔플
            perm  = torch.randperm(len(valid))
            valid = valid[perm]

            # 배치 단위 학습
            for start in range(0, len(valid), batch_size):
                batch_idx = valid[start : start + batch_size]  # (B,)
                B = len(batch_idx)

                # feat_O: (B, 1, F) → (B, N, F) by expand
                feat_O = X_s[batch_idx].unsqueeze(1).expand(B, N, F)   # (B, N, F)
                feat_D = X_s.unsqueeze(0).expand(B, N, F)               # (B, N, F)
                log_d  = X_d[batch_idx].unsqueeze(-1)                   # (B, N, 1)

                feat = torch.cat([feat_O, feat_D, log_d], dim=-1)       # (B, N, 2F+1)
                feat_flat = feat.view(B * N, -1)                        # (B*N, 2F+1)

                flows = X_od[batch_idx]                                  # (B, N)

                optimizer.zero_grad()
                logits_flat = model(feat_flat)                          # (B*N,)
                logits = logits_flat.view(B, N)                         # (B, N)

                # row-wise multinomial NLL, B로 나눠서 gradient 안정화
                log_p = torch.nn.functional.log_softmax(logits, dim=1)  # (B, N)
                loss  = -(flows * log_p).sum() / B

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient explosion 방지
                optimizer.step()

                total_loss += loss.item()
                n_origins  += B

        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_origins, 1)
        print(f"Epoch {epoch:2d}/{n_epochs}  Loss={avg_loss:.4f}  "
              f"origins={n_origins}  time={elapsed:.1f}s")
        
def predict_od(dg_model: DeepGravityFFN,
               gen_model,
               sample: dict,
               use_lgbm: bool,
               device: torch.device,
               row_chunk: int = 64) -> np.ndarray:
    X_static_raw = sample['X_static_raw'].float()   # (N, F) CPU tensor
    X_dist_raw = torch.expm1(sample['X_dist'].float())      # (N, N) CPU tensor (log-scaled)
    N, F = X_static_raw.shape

    # === 총 발생량 예측 ===
    if use_lgbm:
        O_pred = np.maximum(gen_model.predict(X_static_raw.numpy(), device), 0)  # (N,)
        O_pred_t = torch.tensor(O_pred, dtype=torch.float32)             # CPU
    else:
        with torch.no_grad():
            O_pred_t = gen_model(
                X_static_raw.to(device)
            ).clamp(min=0).cpu()                                         # (N,) CPU
        O_pred = O_pred_t.numpy()

    # === 분포 예측: row-chunk 방식으로 VRAM 절약 ===
    X_s = X_static_raw.to(device)     # (N, F)
    X_d = X_dist_raw.to(device)       # (N, N) raw distance (학습과 동일)

    logits_rows = []
    with torch.no_grad():
        for start in range(0, N, row_chunk):
            end = min(start + row_chunk, N)
            B   = end - start
            feat_O = X_s[start:end].unsqueeze(1).expand(B, N, F)  # (B, N, F)
            feat_D = X_s.unsqueeze(0).expand(B, N, F)             # (B, N, F)
            log_d  = X_d[start:end].unsqueeze(-1)                  # (B, N, 1)
            feat   = torch.cat([feat_O, feat_D, log_d], dim=-1)    # (B, N, 2F+1)
            logits_chunk = dg_model(feat.view(B * N, -1)).view(B, N)
            logits_rows.append(logits_chunk.cpu())

    logits = torch.cat(logits_rows, dim=0)                          # (N, N) CPU
    log_p  = torch.nn.functional.log_softmax(logits, dim=1)
    p      = torch.exp(log_p)                                        # (N, N)

    T_pred = (p * O_pred_t.unsqueeze(1)).numpy()                    # (N, N)
    return T_pred


def _eval_one(dg_model, gen_model, base_data, use_lgbm, device,
              year_label, city_name, task, split_name, mask_indices, merge_events):
    """sample 1개 평가 → dict 반환"""
    try:
        sample = apply_merge_events(base_data, mask_indices, merge_events)

        # ── 첫 번째 샘플에서만 진단 출력 ──────────────────────────────────
        if not getattr(_eval_one, '_diagnosed', False):
            _eval_one._diagnosed = True
            xs = sample['X_static_raw'].float()
            xd = sample['X_dist'].float()
            print(f"\n[DIAG] X_static_raw: shape={tuple(xs.shape)}  "
                  f"min={xs.min():.3f}  max={xs.max():.3f}  "
                  f"nan={torch.isnan(xs).any().item()}")
            print(f"[DIAG] X_dist:       shape={tuple(xd.shape)}  "
                  f"min={xd.min():.3f}  max={xd.max():.3f}  "
                  f"nan={torch.isnan(xd).any().item()}")
            # 모델 weight NaN 체크
            nan_params = [n for n, p in dg_model.named_parameters()
                          if torch.isnan(p).any()]
            print(f"[DIAG] Model NaN weights: {nan_params if nan_params else 'none'}")
            # 첫 번째 레이어 weight 범위
            first_w = next(dg_model.parameters())
            print(f"[DIAG] First layer weight: "
                  f"min={first_w.min():.4f}  max={first_w.max():.4f}")

        T_pred = predict_od(dg_model, gen_model, sample, use_lgbm, device)

        # T_pred NaN 진단 (처음 발견 시)
        if not getattr(_eval_one, '_pred_diagnosed', False) and np.isnan(T_pred).any():
            _eval_one._pred_diagnosed = True
            xs = sample['X_static_raw'].float()
            print(f"\n[DIAG] T_pred has NaN! ({np.isnan(T_pred).sum()} / {T_pred.size} entries)")
            print(f"[DIAG] O_pred range: checking LGBM output...")
            if use_lgbm:
                o = np.maximum(gen_model.predict(xs.numpy(), device), 0)
                print(f"[DIAG] O_pred: min={o.min():.2f}  max={o.max():.2f}  "
                      f"nan={np.isnan(o).any()}")
            # logit 범위 직접 체크
            with torch.no_grad():
                x_s2 = xs.to(device)
                B, N, F = 1, xs.shape[0], xs.shape[1]
                feat_O = x_s2[0:1].unsqueeze(1).expand(1, N, F)
                feat_D = x_s2.unsqueeze(0).expand(1, N, F)
                log_d  = sample['X_dist'].float().to(device)[0:1].unsqueeze(-1)
                feat   = torch.cat([feat_O, feat_D, log_d], dim=-1).view(N, -1)
                logit1 = dg_model(feat)
            print(f"[DIAG] logit[0] row: min={logit1.min():.2f}  "
                  f"max={logit1.max():.2f}  nan={torch.isnan(logit1).any()}")

        y_od = sample['y_OD_raw'].numpy()
        eval_idx = np.array(mask_indices)
        N = y_od.shape[0]
        mask2d = np.zeros((N, N), dtype=bool)
        mask2d[:, eval_idx] = True
        mask2d[eval_idx, :] = True

        y_eval    = y_od[mask2d]
        y_pred_ev = T_pred[mask2d]

        rmse  = float(np.sqrt(np.mean((y_eval - y_pred_ev) ** 2)))
        cpc   = cpc_score(y_eval, y_pred_ev)
        prmse = rmse / float(np.mean(y_eval)) if np.mean(y_eval) > 0 else 0.0

        return {'year': year_label, 'city': city_name, 'task': task,
                'split': split_name, 'rmse': rmse, 'cpc': cpc, 'prmse': prmse}
    except Exception as e:
        import traceback
        print(f"  [WARN] {city_name} task={task}: {e}")
        if not getattr(_eval_one, '_traced', False):
            _eval_one._traced = True
            traceback.print_exc()
        return None


def evaluate_and_report(dg_model, gen_model, base_data, meta_dict,
                        use_lgbm, year_label, split_name,
                        device, n_workers=4):
    """
    GPU: 시퀴셌셜 (스레드가 CUDA 감려로 오히려 느려짐)
    CPU: ThreadPoolExecutor로 apply_merge_events CPU 병렬화
    """
    is_cuda = device.type == 'cuda'

    job_args = []
    for task in range(5):
        for city_name, task_list in meta_dict.items():
            for meta in task_list[task]:
                job_args.append((
                    dg_model, gen_model, base_data, use_lgbm, device,
                    year_label, city_name, task, split_name,
                    meta['mask_indices'], meta['merge_events']
                ))

    total = len(job_args)
    mode_str = 'sequential(GPU)' if is_cuda else f'threads={n_workers}'
    print(f"  [{year_label}/{split_name}] {total}개 샘플 평가 ({mode_str})...")

    dg_model.eval()
    if is_cuda:
        # CUDA 첫 번째 kernel 컴파일 워밍업 (첫 sample 느린 문제 방지)
        try:
            dummy = torch.zeros(1, 37, device=device)
            with torch.no_grad():
                dg_model(dummy)
        except Exception:
            pass

        # tqdm.auto: Colab/Jupyter 환경 자동 감지
        try:
            from tqdm.auto import tqdm
            it = tqdm(job_args, desc=f"[{year_label}/{split_name}]", ncols=80)
            use_tqdm = True
        except ImportError:
            it = job_args
            use_tqdm = False

        results = []
        for idx, a in enumerate(it):
            results.append(_eval_one(*a))
            # tqdm 없을 때 50개마다 수동 출력
            if not use_tqdm and (idx + 1) % 50 == 0:
                print(f"  [{year_label}/{split_name}] {idx+1}/{total}...", flush=True)
    else:
        def _wrap(a):
            return _eval_one(*a)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_wrap, job_args))

    records = [r for r in results if r is not None]
    print(f"  [{year_label}/{split_name}] 완료: {len(records)}/{total}")
    return records


def summarize_results(records, group_keys, label):
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in records:
        key = tuple(rec[k] for k in group_keys)
        for m in ('rmse', 'cpc', 'prmse'):
            grouped[key][m].append(rec[m])
    print(f"\n=== {label} ===")
    for key in sorted(grouped.keys()):
        key_str = ", ".join(f"{k}={v}" for k, v in zip(group_keys, key))
        n = len(grouped[key]['cpc'])
        cpc_m, cpc_s   = np.mean(grouped[key]['cpc']),   np.std(grouped[key]['cpc'])
        rmse_m, rmse_s = np.mean(grouped[key]['rmse']),  np.std(grouped[key]['rmse'])
        pr_m, pr_s     = np.mean(grouped[key]['prmse']), np.std(grouped[key]['prmse'])
        print(f"[{key_str}] (n={n}) "
              f"CPC={cpc_m:.4f}±{cpc_s:.4f}  "
              f"RMSE={rmse_m:.4f}±{rmse_s:.4f}  "
              f"%RMSE={pr_m:.4f}±{pr_s:.4f}")