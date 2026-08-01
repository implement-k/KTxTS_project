"""
실행 방법:
    python train_and_val.py --generation_model lgbm   # LGBM으로 O 예측
    python train_and_val.py --generation_model ffn    # FFN으로 O 예측

구조:
    1. 총 발생량(O_i) 예측: LGBM 또는 FFN (--generation_model)
    2. 분포 예측(Distribution): Deep Gravity FFN
       input = [feat_O | feat_D | dist] → logit → Softmax → T_ij = p_ij * O_i
    3. 학습: Multinomial NLL
    4. 평가: fixed_eval 기반 CPC/RMSE/%RMSE
"""

import os, sys, argparse, warnings, torch
import numpy as np
import torch.optim as optim

from dataset import ODDataset
from model import GenerationModel, DeepGravityFFN
from train_eval import train, evaluate_and_report, summarize_results, eval_generation_model

warnings.filterwarnings('ignore')

# 경로 설정
SRC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = os.path.dirname(SRC_PATH)
EVAL_PATH = os.path.join(SRC_PATH, 'evaluation')
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, EVAL_PATH)


# argument parsing
# 기본값은 deep gravity 코드값 이용
def parse_args():
    parser = argparse.ArgumentParser(description='DeepGravity — KT 데이터')
    parser.add_argument('--mode', default='train', choices=['train', 'val', 'test'],
                        help='train: 학습, val: 검증, test: 테스트')
    parser.add_argument('--generation_model', default='lgbm', choices=['lgbm', 'ffn'],
                        help='총 발생량 예측 모델 (lgbm | ffn)')
    parser.add_argument('--batch_size', type=int, default=32,
                            help='origin 배치 크기')
    parser.add_argument('--gen_epochs',   type=int, default=15)
    parser.add_argument('--epochs',   type=int, default=15)
    parser.add_argument('--lr',       type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0.9, 
                            help='SGD momentum (default: 0.9)') 
    parser.add_argument('--hidden',   type=int, default=256)
    parser.add_argument('--dropout',  type=float, default=0.35)
    parser.add_argument('--workers',  type=int, default=4)
    parser.add_argument('--imputation', default='zero', choices=['zero', 'mean'])
    parser.add_argument('--device',   default='gpu', choices=['cpu', 'gpu'],)
    
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                            help='모델 저장/로드 디렉토리')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    use_lgbm = (args.generation_model == 'lgbm')
    year_labels: list[str] = ['2019', '2023']
    print(f"\n[설정] generation_model={args.generation_model}  device={device}")
    
    # === 데이터셋 로드 ===
    print("\nI: 1. 데이터셋 로드...")
    datasets: list[ODDataset] = []
    for year in year_labels:
        dataset = ODDataset(year=year, imputation=args.imputation, use_raw_static=False)
        datasets.append(dataset)
        
    F = datasets[0].X_static_raw.shape[1]
    
    # 2019 vs 2023 피처 차원 불일치 확인
    if F != datasets[1].X_static_raw.shape[1]:
        raise ValueError(f"E: Feature dimension mismatch: (2019){datasets[0].X_static_raw.shape[1]} vs (2023){datasets[1].X_static_raw.shape[1]}")
            
    
    # === 모델 선언 ===
    # 피처 차원 확인
    dim_input = 2 * F + 1  # [feat_O | feat_D | log_dist]
    print(f"I: 피처 차원 F={F}  분포모델 입력={dim_input}")
    gen_model = GenerationModel(use_lgbm=use_lgbm, dim_input=F)
    dg_model = DeepGravityFFN(dim_input, dim_hidden=args.hidden, dropout_p=args.dropout).to(device) # deep gravity에는 없는 모델(통행량 생성을 futeure work로 명시했음.)
    
    # 체크포인트 디렉토리
    ckpt_dir = os.path.join(ROOT_PATH, args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_dg_path  = os.path.join(ckpt_dir, f'dg_model_{args.generation_model}.pt')
    ckpt_gen_path = os.path.join(ckpt_dir, f'gen_model_{args.generation_model}.pkl')

    # 학습
    if args.mode == 'train':
        # 생성 모델 학습
        if use_lgbm:
            X_static_train_all = np.concatenate([ds.X_static_raw[ds.train_indices] for ds in datasets], axis=0)
        else:
            X_static_train_all = np.concatenate([ds.X_static[ds.train_indices] for ds in datasets], axis=0)
        O_train_all = np.concatenate([ds.y_o[ds.train_indices] for ds in datasets], axis=0)
        
        print("\nI: 2. 생성 모델 학습 시작")
        gen_model.fit(X_static_train_all, O_train_all, epochs=args.gen_epochs, device=device)
        print("I: 생성 모델 학습 완료")
        
        eval_generation_model(gen_model, datasets, use_lgbm, device)

        # 분포 모델 학습
        print("\nI: 3. DeepGravity 분포 모델 학습 시작")
        optimizer = optim.RMSprop(dg_model.parameters(), lr=args.lr, momentum=args.momentum)
        data_list = [
            {
                'X_static': ds.X_static,
                'X_dist': ds.X_dist,
                'X_OD': ds.X_OD,
                'train_idx': ds.train_indices,
            }
            for ds in datasets
        ]
        train(dg_model, optimizer, data_list, device,
            n_epochs=args.epochs, batch_size=args.batch_size)
        print("I: 분포 모델 학습 완료")

        # 모델 저장
        import pickle
        torch.save(dg_model.state_dict(), ckpt_dg_path)
        with open(ckpt_gen_path, 'wb') as f:
            pickle.dump(gen_model, f)
        print(f"I: 모델 저장 완료 → {ckpt_dir}")

    # 평가 (val / test)
    else:
        # 모델 로드
        import pickle
        if not os.path.exists(ckpt_dg_path) or not os.path.exists(ckpt_gen_path):
            raise FileNotFoundError(
                f"E: 저장된 모델이 없습니다. 먼저 --mode train 으로 학습하세요.\n"
                f"  dg : {ckpt_dg_path}\n  gen: {ckpt_gen_path}"
            )
        dg_model.load_state_dict(torch.load(ckpt_dg_path, map_location=device))
        dg_model.eval()
        with open(ckpt_gen_path, 'rb') as f:
            gen_model = pickle.load(f)
        print(f"I: 모델 로드 완료 ← {ckpt_dir}")

        print(f"\nI: 평가 시작 (mode={args.mode})")
        fixed_eval_dir = os.path.join(ROOT_PATH, 'dataset', 'fixed_eval')
        all_records = []

        for year in year_labels:
            base_data_path = os.path.join(fixed_eval_dir, f'base_data_{year}.pt')
            if not os.path.exists(base_data_path):
                print(f"W: [SKIP] base_data_{year}.pt not found")
                continue
            base_data = torch.load(base_data_path, weights_only=False)
            
            meta_data_path = os.path.join(fixed_eval_dir, f'fixed_{args.mode}_meta_{year}.pt')
            if not os.path.exists(meta_data_path):
                print(f"W: [SKIP] fixed_{args.mode}_meta_{year}.pt not found")
                continue
            meta_dict = torch.load(meta_data_path, weights_only=False)
            records = evaluate_and_report(
                dg_model, gen_model, base_data, meta_dict,
                use_lgbm, year, args.mode, device, n_workers=args.workers
            )
            all_records.extend(records)

        # 결과 요약
        print("\n===연도별===")
        summarize_results(all_records, ['year', 'split', 'task'], "task별 종합 성능")
        summarize_results(all_records, ['year', 'split', 'city'], "도시별 성능")

        print("\n===연도 종합===")
        summarize_results(all_records, ['split', 'task'], "task별 종합 성능 (모든 도시)")
        summarize_results(all_records, ['split', 'city'], "도시별 성능 (모든 task)")

        print("\n===총 종합===")
        summarize_results(all_records, ['split'], "전체 종합")


if __name__ == '__main__':
    main()
