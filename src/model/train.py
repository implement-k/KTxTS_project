import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_CONFIG

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib

from dataset import ODDataset
from models import DeepGravity, SpatialODMAE1, SpatialODMAE5, LGBMModel
from tqdm import tqdm
from model_test import test_dl_model
from loss import LOSS_REGISTRY
from eval_utils import set_seed, eval_mode_for_validation, check_validation_determinism

STATE_PATH = 'training_state.pt'
COMPLETED_PATH = 'completed.json'


def get_git_commit_hash():
    try:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def canonical_json(params):
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def sha256_of_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path, obj):
    """completed.json을 원자적으로 쓴다: 임시 파일에 쓴 뒤 os.replace로 교체(중간에 끊겨도 기존 파일 보존)."""
    d = os.path.dirname(os.path.abspath(path)) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=d, prefix='.completed_tmp_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def main():
    print("Starting Training...")
    '''
    사용법
    python train.py --model mae1 --epochs 50 --batch_size 32 --loss dynamic_weighted_mse \
        --seed 42 --protocol strict [--run-final-test]

    protocol:
      legacy - 기존 test_dataset을 validation/checkpoint 선택/최종 평가에 모두 사용하는 과거 방식.
               historical baseline(CPC 0.5434) 재현 전용이며, 학습 종료 시 항상 최종 test까지 자동 실행.
      strict - test_indices를 training loss/validation에서 완전히 배제. train_indices 내부에서
               group 단위로 떼어낸 validation node(mode='val')로 checkpoint를 선정한다.
               --run-final-test를 명시적으로 줘야만 최종(test_indices) 평가를 수행한다
               (신규 loss 순위 선정에는 test 결과를 쓰지 않기 위함).
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=TRAIN_CONFIG['model_type'], choices=['lgbm', 'deep_gravity', 'mae1', 'mae5'])
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--loss', type=str, default='dynamic_weighted_mse', choices=list(LOSS_REGISTRY.keys()))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--loss-params', type=str, default='{}',
                         help='선택한 loss 생성자에 전달할 키워드 인자 JSON 문자열 (예: \'{"lambda_log": 0.5}\')')
    parser.add_argument('--protocol', type=str, default='legacy', choices=['legacy', 'strict'])
    parser.add_argument('--run-final-test', action='store_true',
                         help='strict protocol에서 test_indices 최종 평가를 명시적으로 실행(후보 확정 후에만 사용)')
    parser.add_argument('--val-fraction', type=float, default=0.10)
    parser.add_argument('--val-seed', type=int, default=42)
    parser.add_argument('--stop-epoch', type=int, default=None,
                         help='--epochs(planned total)는 그대로 두고 실제로는 이 epoch에서 조기 종료. '
                              'scheduler/mask curriculum은 --epochs 기준(예: 50)을 그대로 쓰면서 '
                              '15 epoch만 실행하는 식의 screening에 사용(Stage 2).')
    parser.add_argument('--smoke-steps', type=int, default=None,
                         help='지정하면 epoch 0에서 이 스텝 수만큼만 학습 후 강제로 1회 validation을 '
                              '수행하고 종료(Stage 1 runtime smoke 전용). gradient norm/clip 비율/'
                              'loss 내부 term 등 진단 정보를 smoke_diagnostics.json으로 남긴다.')
    args = parser.parse_args()

    set_seed(args.seed)

    # 데이터셋 로드
    channel = 5 if args.model == 'mae5' else 1
    is_log = args.model in ['mae1', 'mae5', 'deep_gravity']
    train_dataset = ODDataset(mode='train', channel=channel, isLogScale=is_log,
                               protocol=args.protocol, val_fraction=args.val_fraction, val_seed=args.val_seed)
    test_dataset = ODDataset(mode='test', channel=channel, isLogScale=is_log,
                              protocol=args.protocol, val_fraction=args.val_fraction, val_seed=args.val_seed)
    val_dataset = None
    if args.protocol == 'strict':
        val_dataset = ODDataset(mode='val', channel=channel, isLogScale=is_log,
                                 protocol='strict', val_fraction=args.val_fraction, val_seed=args.val_seed)

    if args.model in ['mae1', 'mae5', 'deep_gravity']:
        train_dl_model(args, train_dataset, val_dataset, test_dataset)
    elif args.model in ['lgbm']:
        train_tabular_model(args, test_dataset)


def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0: return 0.0
    return numerator / denominator


def build_run_identity(args):
    """resume 시 이전 실행과 현재 실행의 조건이 완전히 일치하는지 비교하기 위한 식별자."""
    loss_params = json.loads(args.loss_params)
    if args.loss in ('weighted_mse_fixed',) and 'alpha' not in loss_params:
        loss_params['alpha'] = 1.5
    return {
        'protocol': args.protocol,
        'loss': args.loss,
        'canonical_loss_params': canonical_json(loss_params),
        'seed': args.seed,
        'planned_epochs': args.epochs,
        'batch_size': args.batch_size,
        'model': args.model,
        'git_commit': get_git_commit_hash(),
    }, loss_params


def save_training_state(path, model, optimizer, scheduler, epoch, best_val_metric, identity):
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'current_epoch': epoch,
        'best_val_metric': best_val_metric,
        'rng_random': random.getstate(),
        'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        **identity,
    }
    # 저장도 원자적으로(임시 파일 -> rename)
    tmp_path = path + '.tmp'
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def try_resume(path, identity, model, optimizer, scheduler, device):
    """
    metadata가 현재 실행 조건과 완전히 일치할 때만 resume한다. 완료 판단(=resume 여부)은
    checkpoint/CSV 존재만으로 하지 않고, 이 training_state.pt의 메타데이터를 정확히 비교한다.
    """
    if not os.path.exists(path):
        return 0, float('inf'), False

    state = torch.load(path, map_location=device, weights_only=False)
    mismatch = []
    for k in identity:
        if state.get(k) != identity[k]:
            mismatch.append((k, state.get(k), identity[k]))

    if mismatch:
        print(f"[resume] training_state.pt 발견했지만 조건이 다름(처음부터 시작): {mismatch}")
        return 0, float('inf'), False

    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    scheduler.load_state_dict(state['scheduler_state'])
    random.setstate(state['rng_random'])
    np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'].cpu() if torch.is_tensor(state['rng_torch']) else state['rng_torch'])
    if torch.cuda.is_available() and state.get('rng_cuda') is not None:
        torch.cuda.set_rng_state_all(state['rng_cuda'])

    start_epoch = state['current_epoch'] + 1
    best_val_metric = state['best_val_metric']
    print(f"[resume] training_state.pt 조건 일치 — epoch {start_epoch}부터 재개 (best_val_metric={best_val_metric})")
    return start_epoch, best_val_metric, True


def validate_csv_schema(csv_path):
    """completed.json에 남길 CSV 스키마 검증 결과. eval_utils.OFFICIAL_CATEGORIES와 일치하는지 확인."""
    try:
        import pandas as pd
        from eval_utils import OFFICIAL_CATEGORIES
        df = pd.read_csv(csv_path)
        required_cols = {'category', 'evaluation_type', 'rmse', 'mae', 'cpc', 'n_samples'}
        missing_cols = required_cols - set(df.columns)
        official_rows = df[df['evaluation_type'] == 'official_masked']['category'].tolist()
        categories_ok = official_rows == OFFICIAL_CATEGORIES
        return {
            'ok': (not missing_cols) and categories_ok,
            'missing_columns': sorted(missing_cols),
            'categories_match': categories_ok,
            'found_categories': official_rows,
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def train_dl_model(args, train_dataset, val_dataset, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.protocol == 'strict' and args.model != 'mae1':
        print(f"[경고] strict protocol은 mae1 기준으로 설계·검증되었습니다(model={args.model}). "
              f"deep_gravity/mae5 경로에서는 val_dataset 처리가 완전히 검증되지 않았을 수 있습니다.")

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) if val_dataset is not None else None

    if args.model == 'mae5':
        model = SpatialODMAE5(num_nodes=train_dataset.num_nodes, num_features=train_dataset.X_static.shape[1]).to(device)
    elif args.model == 'mae1':
        model = SpatialODMAE1(num_nodes=train_dataset.num_nodes, num_features=train_dataset.X_static.shape[1]).to(device)
    else:
        model = DeepGravity(num_features=train_dataset.X_static.shape[1]).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    identity, loss_params = build_run_identity(args)
    criterion = LOSS_REGISTRY[args.loss](**loss_params).to(device)

    # === MPS/CUDA validation determinism 확인 (작은 테스트, 학습에 영향 없음) ===
    try:
        probe_batch = next(iter(val_loader if val_loader is not None else test_loader))
        x_s = probe_batch['X_static'].to(device)
        x_d = probe_batch['X_dist'].to(device)
        m = probe_batch['mask'].to(device)
        x_o = probe_batch['X_OD_masked'].to(device)

        def _forward():
            with eval_mode_for_validation(model, device=device):
                if args.model == 'deep_gravity':
                    return model(x_s, x_d)
                return model(x_s, x_o, x_d, m)

        det = check_validation_determinism(_forward, device=device)
        print(f"[determinism check] device={det['device']} is_deterministic={det['is_deterministic']} "
              f"max_abs_diff={det['max_abs_diff']:.3e}")
    except Exception as e:
        print(f"[determinism check] 건너뜀(probe 실패): {e}")

    min_mask = TRAIN_CONFIG['min_mask_size']
    max_mask = TRAIN_CONFIG['max_mask_size']

    best_model_path = f'best_model_{args.model}_{args.loss}_seed{args.seed}_{args.protocol}.pth'
    max_norm = 1.0

    # Stage 1 runtime smoke 진단용 누적 통계(--smoke-steps 지정 시에만 의미 있음, 그 외엔 그냥 참고용)
    grad_norms = []
    clipped_steps = 0
    total_steps_done = 0
    loss_term_samples = []

    end_epoch = min(args.epochs, args.stop_epoch) if args.stop_epoch is not None else args.epochs
    start_epoch, best_val_metric, resumed = try_resume(STATE_PATH, identity, model, optimizer, scheduler, device)
    smoke_stopped_early = False
    if resumed and start_epoch >= end_epoch:
        print(f"[resume] 이미 계획된 {end_epoch} epoch(stop_epoch 기준)을 모두 마쳤습니다. 학습을 건너뜁니다.")
    else:
        for epoch in range(start_epoch, end_epoch):
            # masking size 결정(min_mask ~ current_mask_size) 랜덤으로 선택
            # 주의: planned_total_epochs=args.epochs 기준으로 커리큘럼을 계산 — screening(stop_epoch<epochs)
            # 실행이어도 50 epoch 기준 스케줄을 그대로 쓴다(오케스트레이터가 --epochs는 항상 계획된
            # 총 epoch로 고정하고 --stop-epoch로만 조기 종료를 지시함).
            progress = epoch / max(1, args.epochs - 1)
            current_mask_size = int(min_mask + (max_mask - min_mask) * progress)
            train_dataset.max_mask_size = current_mask_size

            # dynamic_weighted_mse: git a17b361에서 복원한 alpha 10.0->1.0 스케줄을 여기서 적용.
            # 다른 loss(weighted_mse_fixed 포함)는 alpha가 있어도 건드리지 않는다(고정값 유지가 설계 의도).
            if args.loss == 'dynamic_weighted_mse':
                criterion.alpha = max(1.0, 10.0 * (1.0 - progress))

            model.train()
            train_loss = 0

            if args.model in ['mae1', 'mae5']:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Mask: {current_mask_size}]")
                for batch in pbar:
                    x_static = batch['X_static'].to(device)
                    x_dist = batch['X_dist'].to(device)
                    mask = batch['mask'].to(device)

                    x_od_masked = batch['X_OD_masked'].to(device)
                    y_od = batch['y_OD'].to(device)

                    optimizer.zero_grad()
                    pred = model(x_static, x_od_masked, x_dist, mask)
                    mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)

                    if args.model == 'mae5':
                        mask_expanded = mask_2d.unsqueeze(-1).expand_as(y_od)
                        loss = criterion(pred, y_od, mask_expanded)
                    else:
                        loss = criterion(pred, y_od, mask_2d)

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    optimizer.step()
                    scheduler.step()

                    gn = float(grad_norm)
                    grad_norms.append(gn)
                    if gn > max_norm:
                        clipped_steps += 1
                    for attr in ('last_log_term', 'last_real_term', 'last_mse_term', 'last_cpc_term'):
                        v = getattr(criterion, attr, None)
                        if v is not None:
                            loss_term_samples.append({'step': total_steps_done, attr: float(v)})
                    total_steps_done += 1

                    train_loss += loss.item()
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({'loss': loss.item(), 'lr': f"{current_lr:.1e}", 'gn': f"{gn:.2f}"})

                    if args.smoke_steps is not None and total_steps_done >= args.smoke_steps:
                        smoke_stopped_early = True
                        break
                print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f} "
                      f"(누적 스텝 {total_steps_done})")

            elif args.model == 'deep_gravity':
                is_train_node = torch.ones(train_dataset.num_nodes, dtype=torch.bool, device=device)
                is_train_node[train_dataset.test_indices] = False
                train_mask_2d = is_train_node.unsqueeze(0) & is_train_node.unsqueeze(1)
                train_mask_2d = train_mask_2d.unsqueeze(0)

                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
                for batch in pbar:
                    x_static = batch['X_static'][0:1].to(device)
                    x_dist = batch['X_dist'][0:1].to(device)
                    y_od = batch['y_OD'][0:1].to(device)

                    optimizer.zero_grad()
                    pred = model(x_static, x_dist)
                    loss = criterion(pred, y_od, train_mask_2d)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                    train_loss += loss.item()
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({'loss': loss.item(), 'lr': f"{current_lr:.1e}"})

                print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")

            # === Validation (2 Epoch마다 수행, 단 --smoke-steps로 조기 종료한 경우 강제로 1회 수행) ===
            # legacy: test_dataset을 그대로 validation에 사용(historical 재현 목적, Codex가 지적한
            #         "checkpoint 선택에도 test set을 쓰는" 문제를 그대로 유지 — 의도적).
            # strict: val_dataset(strict_val_indices, test_indices와 무관)만 사용.
            eval_loader = val_loader if val_loader is not None else test_loader
            if epoch % 2 == 1 or epoch == end_epoch - 1 or smoke_stopped_early:
                with eval_mode_for_validation(model, device=device), torch.no_grad():
                    val_batch = next(iter(eval_loader))
                    x_s = val_batch['X_static'].to(device)
                    x_d = val_batch['X_dist'].to(device)
                    m = val_batch['mask'].to(device)
                    x_o = val_batch['X_OD_masked'].to(device)
                    y_o = val_batch['y_OD'].to(device)

                    if args.model == 'deep_gravity':
                        v_pred = model(x_s, x_d)
                    else:
                        v_pred = model(x_s, x_o, x_d, m)

                    m2d = m.unsqueeze(1) | m.unsqueeze(2)

                    if args.model == 'mae5':
                        m_exp = m2d.unsqueeze(-1).expand_as(y_o)
                        v_loss = criterion(v_pred, y_o, m_exp).item()
                        p_real = np.maximum(torch.expm1(v_pred[m_exp]).cpu().numpy(), 0)
                        y_real = torch.expm1(y_o[m_exp]).cpu().numpy()
                    else:
                        v_loss = criterion(v_pred, y_o, m2d).item()
                        p_real = np.maximum(torch.expm1(v_pred[m2d]).cpu().numpy(), 0)
                        y_real = torch.expm1(y_o[m2d]).cpu().numpy()

                    rmse = np.sqrt(np.mean((y_real - p_real)**2))
                    cpc = cpc_score(y_real, p_real)
                print(f"  ➜ [Val/{args.protocol}] Loss: {v_loss:.4f} | RMSE: {rmse:.2f} | CPC: {cpc:.4f}")

                if rmse < best_val_metric:
                    best_val_metric = rmse
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  ➜ [Checkpoint] Best model saved! (RMSE: {rmse:.2f})")

                model.train()

            # === epoch마다 full training state 저장(resume용) ===
            save_training_state(STATE_PATH, model, optimizer, scheduler, epoch, best_val_metric, identity)

            if smoke_stopped_early:
                break

        print("Training finished.")

    if args.smoke_steps is not None:
        gn_arr = np.array(grad_norms) if grad_norms else np.array([0.0])
        diagnostics = {
            'loss': args.loss,
            'protocol': args.protocol,
            'total_steps_done': total_steps_done,
            'requested_smoke_steps': args.smoke_steps,
            'grad_norm_mean': float(gn_arr.mean()),
            'grad_norm_max': float(gn_arr.max()),
            'grad_norm_min': float(gn_arr.min()),
            'clip_max_norm': max_norm,
            'clipped_step_ratio': (clipped_steps / total_steps_done) if total_steps_done else None,
            'nan_or_inf_grad_norm': bool(np.isnan(gn_arr).any() or np.isinf(gn_arr).any()),
            'loss_term_samples': loss_term_samples[:20],  # 앞부분만 샘플로 기록(용량 절약)
        }
        atomic_write_json('smoke_diagnostics.json', diagnostics)
        print(f"smoke_diagnostics.json 작성 완료: {diagnostics}")

    # completed.json은 "계획된 전체 epoch(args.epochs)까지 실제로 도달한 완전한 실행"에만 쓴다.
    # --stop-epoch로 조기 종료한 screening 실행은 여기 해당하지 않음(training_state.pt의
    # current_epoch로 오케스트레이터가 "어디까지 진행됐는지"를 별도로 판단한다) — completed.json과
    # 부분 실행 상태가 섞이면 Stage 2(15 epoch)가 Stage 3(50 epoch)를 "이미 완료"로 오판하게 됨.
    reached_planned_epochs = (args.stop_epoch is None) or (end_epoch >= args.epochs)

    if args.smoke_steps is not None:
        # smoke 실행의 "완료"는 completed.json이 아니라 smoke_diagnostics.json 존재로 판단한다
        # (args.epochs를 그대로 기록하면 "50 epoch를 다 돌았다"는 오해를 줄 수 있어 분리함).
        print("[smoke] completed.json은 쓰지 않음 — smoke_diagnostics.json이 완료 신호.")
    elif not reached_planned_epochs:
        print(f"[screening] stop_epoch={args.stop_epoch}로 조기 종료 — completed.json은 쓰지 않음 "
              f"(training_state.pt의 current_epoch로 진행 상태 판단, 추후 --stop-epoch 없이 재실행하면 이어서 진행).")
    else:
        # === 최종 평가: legacy는 항상, strict는 --run-final-test를 명시했을 때만 ===
        run_final_test = (args.protocol == 'legacy') or args.run_final_test
        if run_final_test:
            test_dl_model(args, test_dataset)
            csv_path = f'results_{args.model}_{args.loss}_seed{args.seed}_{args.protocol}.csv'
            schema_result = validate_csv_schema(csv_path)
        else:
            print(f"[protocol=strict] --run-final-test가 없어 최종(test_indices) 평가를 건너뜁니다 "
                  f"(후보 확정 전에는 test를 건드리지 않기 위함).")
            csv_path = None
            schema_result = {'ok': None, 'skipped': True}

        completed = {
            'returncode': 0,
            'protocol': args.protocol,
            'git_commit': identity['git_commit'],
            'loss': args.loss,
            'canonical_loss_params': identity['canonical_loss_params'],
            'seed': args.seed,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'model': args.model,
            'checkpoint_path': best_model_path,
            'checkpoint_sha256': sha256_of_file(best_model_path),
            'ran_final_test': run_final_test,
            'csv_path': csv_path,
            'csv_schema_validation': schema_result,
        }
        atomic_write_json(COMPLETED_PATH, completed)
        print(f"completed.json 작성 완료: {COMPLETED_PATH}")


def train_tabular_model(args, test_dataset):
    X_OD_real = test_dataset.X_OD
    X_dist_real = test_dataset.X_dist
    X_static = test_dataset.X_static
    num_nodes = test_dataset.num_nodes

    O_idx, D_idx = np.indices((num_nodes, num_nodes))
    O_idx = O_idx.flatten()
    D_idx = D_idx.flatten()

    y = X_OD_real.flatten()
    dist = X_dist_real.flatten()

    if args.model == 'lgbm':
        model = LGBMModel()

        O_stat_feat = X_static[O_idx]
        D_stat_feat = X_static[D_idx]
        X_tabular = np.column_stack([dist, O_stat_feat, D_stat_feat])

        test_cities = set(test_dataset.test_indices)
        is_test = np.array([(o in test_cities) or (d in test_cities) for o, d in zip(O_idx, D_idx)])
        is_train = ~is_test

        print(f"Fitting {args.model} Model (this might take a while)...")
        model.fit(X_tabular[is_train], y[is_train])

    save_path = f'best_model_{args.model}.pkl'
    joblib.dump(model, save_path)
    print(f"Training finished. Model saved to {save_path}")


if __name__ == '__main__':
    main()
