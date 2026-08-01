
from model import DeepGravityFFN
import torch, time, os, sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm

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
               row_chunk: int = 256) -> np.ndarray:  # 64 -> 256으로 증가 (VRAM 여유 있으면 더 키워도 됨)
    X_static_raw = sample['X_static_raw'].float()
    X_static_norm = sample['X_static'].float()
    X_dist_raw = sample['X_dist_raw'].float()
    N, F = X_static_norm.shape

    if use_lgbm:
        O_pred = np.maximum(gen_model.predict(X_static_raw.numpy(), device), 0)
        O_pred_t = torch.tensor(O_pred, dtype=torch.float32)
    else:
        with torch.no_grad():
            O_pred_t = gen_model(X_static_norm.to(device)).clamp(min=0).cpu()
        O_pred = O_pred_t.numpy()

    X_s = X_static_norm.to(device)
    X_d = X_dist_raw.to(device)

    # feat_D는 row_chunk 루프 내내 동일하니 딱 한 번만 GPU에 준비
    feat_D_full = X_s.unsqueeze(0)  # (1, N, F), broadcast로 재사용

    logits_rows = []
    with torch.no_grad():
        for start in range(0, N, row_chunk):
            end = min(start + row_chunk, N)
            B = end - start
            feat_O = X_s[start:end].unsqueeze(1).expand(B, N, F)
            feat_D = feat_D_full.expand(B, N, F)
            log_d = X_d[start:end].unsqueeze(-1)
            feat = torch.cat([feat_O, feat_D, log_d], dim=-1)
            logits_chunk = dg_model(feat.view(B * N, -1)).view(B, N)
            logits_rows.append(logits_chunk.cpu())

    logits = torch.cat(logits_rows, dim=0)
    log_p = torch.nn.functional.log_softmax(logits, dim=1)
    p = torch.exp(log_p)
    T_pred = (p * O_pred_t.unsqueeze(1)).numpy()
    return T_pred

def _eval_one(dg_model, gen_model, base_data, use_lgbm, device,
              year_label, city_name, task, split_name, mask_indices, merge_events):
    """sample 1개 평가 -> dict 반환"""
    try:
        sample = apply_merge_events(base_data, mask_indices, merge_events)

        if not getattr(_eval_one, '_diagnosed', False):
            _eval_one._diagnosed = True
            xs = sample['X_static_raw'].float()
            xd = sample['X_dist_raw'].float()
            print(f"\nI: X_static_raw: shape={tuple(xs.shape)}  "
                  f"min={xs.min():.3f}  max={xs.max():.3f}  "
                  f"nan={torch.isnan(xs).any().item()}")
            print(f"I: X_dist: shape={tuple(xd.shape)}  "
                  f"min={xd.min():.3f}  max={xd.max():.3f}  "
                  f"nan={torch.isnan(xd).any().item()}")
            # 모델 weight NaN 체크
            nan_params = [n for n, p in dg_model.named_parameters()
                          if torch.isnan(p).any()]
            print(f"I: Model NaN weights: {nan_params if nan_params else 'none'}")
            # 첫 번째 레이어 weight 범위
            first_w = next(dg_model.parameters())
            print(f"I: First layer weight: "
                  f"min={first_w.min():.4f}  max={first_w.max():.4f}")

        T_pred = predict_od(dg_model, gen_model, sample, use_lgbm, device)

        # T_pred NaN 진단 (처음 발견 시)
        if not getattr(_eval_one, '_pred_diagnosed', False) and np.isnan(T_pred).any():
            _eval_one._pred_diagnosed = True
            print(f"\nW: [{city_name}/task{task}] T_pred has NaN! ({np.isnan(T_pred).sum()} / {T_pred.size})")

            # 1. O_pred 자체 확인 (predict_od와 정확히 같은 경로로)
            if use_lgbm:
                o = np.maximum(gen_model.predict(sample['X_static_raw'].numpy(), device), 0)
            else:
                with torch.no_grad():
                    o = gen_model(sample['X_static'].float().to(device)).clamp(min=0).cpu().numpy()
            print(f"I: O_pred: min={o.min():.2f} max={o.max():.2f} nan={np.isnan(o).any()} inf={np.isinf(o).any()}")

            # 2. X_dist_raw 자체 확인 (이 샘플에서, merge 적용 후)
            xd = sample['X_dist_raw'].numpy()
            print(f"I: X_dist_raw(this sample): min={xd.min():.3f} max={xd.max():.3f} "
                f"nan={np.isnan(xd).any()} inf={np.isinf(xd).any()}")

            # 3. logit을 predict_od와 "정확히 동일한 방식"으로 재현 (정규화된 X_static + X_dist_raw)
            X_s = sample['X_static'].float().to(device)   # 반드시 정규화된 버전
            X_d = sample['X_dist_raw'].float().to(device)
            N, F = X_s.shape
            with torch.no_grad():
                feat_O = X_s[0:1].unsqueeze(1).expand(1, N, F)
                feat_D = X_s.unsqueeze(0).expand(1, N, F)
                log_d = X_d[0:1].unsqueeze(-1)
                feat = torch.cat([feat_O, feat_D, log_d], dim=-1).view(N, -1)
                logit1 = dg_model(feat)
            print(f"I: logit[0] row(정규화 입력): min={logit1.min():.2f} max={logit1.max():.2f} "
                f"nan={torch.isnan(logit1).any().item()}")

            # 4. merge_events 자체를 출력해서, 어떤 병합이 이 샘플의 NaN을 유발했는지 확인
            print(f"I: merge_events for this sample: {merge_events}")

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
        print(f"W: {city_name} task={task}: {e}")
        if not getattr(_eval_one, '_traced', False):
            _eval_one._traced = True
            traceback.print_exc()
        return None


def evaluate_and_report(dg_model, gen_model, base_data, meta_dict,
                        use_lgbm, year_label, split_name,
                        device, n_workers=4):
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
    mode_str = 'cuda' if is_cuda else f'threads={n_workers}'
    print(f"  [{year_label}/{split_name}] {total}개 샘플 평가 ({mode_str})...")

    dg_model.eval()
    if is_cuda:
        # CUDA 메모리 부족 방지용 dummy forward (첫 번째 forward에서 GPU 메모리 할당이 많음)
        try:
            dummy = torch.zeros(1, 37, device=device)
            with torch.no_grad():
                dg_model(dummy)
        except Exception:
            pass

        try:
            it = tqdm(job_args, desc=f"[{year_label}/{split_name}]", ncols=80, mininterval=1.0)
            use_tqdm = True
        except ImportError:
            print("W: tqdm 설치되지 않음. 진행률 표시 없이 평가 wlsgod")
            it = job_args
            use_tqdm = False

        results = []
        for idx, job in enumerate(it):
            results.append(_eval_one(*job))
            
            # tqdm 없을 때 50개마다 수동 출력
            if not use_tqdm and (idx + 1) % 50 == 0:
                print(f"I: [{year_label}/{split_name}] {idx+1}/{total}...", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(lambda a: _eval_one(*a), job_args))

    records = [r for r in results if r is not None]
    print(f"I:  [{year_label}/{split_name}] 완료: {len(records)}/{total}")
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
        
        
def eval_generation_model(gen_model, datasets, use_lgbm, device):
    print("\nI: generation 모델 성능 평가 (val_set)")
    for ds, year_label in zip(datasets, ['2019', '2023']):
        val_idx = ds.val_indices
        if len(val_idx) == 0:
            continue

        if use_lgbm:
            X_val = ds.X_static_raw[val_idx]
            O_pred = np.maximum(gen_model.predict(X_val, device), 0)
        else:
            X_val = ds.X_static[val_idx]  # 정규화된 전체 feature
            O_pred = gen_model.predict(X_val, device)

        O_true = ds.y_o_val[val_idx]

        rmse = float(np.sqrt(np.mean((O_true - O_pred) ** 2)))
        cpc = cpc_score(O_true, O_pred)
        prmse = rmse / float(np.mean(O_true)) if np.mean(O_true) > 0 else 0.0

        print(f"[{year_label} val] n={len(val_idx)}  "
              f"CPC={cpc:.4f}  RMSE={rmse:.2f}  %RMSE={prmse*100:.2f}%")

        # 도시별로 더 세분화해서 보고 싶으면
        for city, city_idx in ds.val_city_indices.items():
            if len(city_idx) == 0:
                continue
            o_true_c = ds.y_o_val[city_idx]
            if use_lgbm:
                o_pred_c = np.maximum(gen_model.predict(ds.X_static_raw[city_idx], device), 0)
            else:
                o_pred_c = gen_model.predict(ds.X_static[city_idx], device)
            rmse_c = float(np.sqrt(np.mean((o_true_c - o_pred_c) ** 2)))
            cpc_c = cpc_score(o_true_c, o_pred_c)
            print(f"    - {city}: CPC={cpc_c:.4f}  RMSE={rmse_c:.2f}  "
                  f"true(mean)={o_true_c.mean():.1f}  pred(mean)={o_pred_c.mean():.1f}")