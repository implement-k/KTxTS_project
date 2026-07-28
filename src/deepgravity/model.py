import lightgbm as lgbm
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim


class GenerationModel:
    """총 발생량(O_i) 예측 모델 래퍼 (LGBM 또는 FFN)"""
    def __init__(self, use_lgbm=True, dim_input=13):
        self.use_lgbm = use_lgbm
        self.dim_input = dim_input

        if self.use_lgbm:
            self.model = lgbm.LGBMRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=5,
                min_child_samples=2, random_state=42, n_jobs=1
            )
        else:
            self.model = GenerationFFN(dim_input=dim_input, dim_hidden=128)

    def fit(self, X_train, y_train, epochs, device):
        if self.use_lgbm:
            print("I: 학습 시작(LGBM)")
            self.model.fit(X_train, y_train) # type: ignore
        else:
            print("I: 학습 시작(FFN)")
            optimizer = optim.Adam(self.model.parameters(), lr=1e-3) # type: ignore
            criterion = nn.MSELoss()
            self.model.train() # type: ignore
            
            x_t = torch.tensor(X_train, dtype=torch.float32, device=device)
            y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.model(x_t) # type: ignore
                loss = criterion(outputs, y_t)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f"  FFN Gen Epoch {epoch+1}/50  Loss={loss.item():.4f}")

    def predict(self, X, device):
        if self.use_lgbm:
            return np.maximum(self.model.predict(X), 0)
        else:
            self.model.eval()
            with torch.no_grad():
                inputs = torch.tensor(X, dtype=torch.float32).to(device)
                outputs = self.model(inputs).cpu().numpy()
            return np.maximum(outputs, 0)



class GenerationFFN(nn.Module):
    """X_static -> 총 발생량(O_i) 예측 FFN"""
    def __init__(self, dim_input: int, dim_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_input, dim_hidden), nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU(),
            nn.Linear(dim_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1).clamp(min=0)



class DeepGravityFFN(nn.Module):
    """
    원 논문(Simini et al., 2021, Nature Communications)의
    NN_MultinomialRegression을 그대로 재현한 버전.

    입력: [feat_O (F) | feat_D (F) | dist (1)]  -> logit (scalar per OD pair)
        - 원 논문 기준 거리(dist)는 raw distance(km), log 변환 없음
    구조: hidden linear layer 15개(dim_hidden 유지 5개 + dim_hidden//2로
          전환 후 10개) + output linear 1개 = 총 16개 linear layer
    학습: Multinomial NLL (= CPC-최적화와 동치)
    예측: softmax(logit) * O_total -> T_hat matrix
    """
    def __init__(self, dim_input: int, dim_hidden: int = 256, dropout_p: float = 0.35):
        super().__init__()
        h = dim_hidden
        h2 = dim_hidden // 2

        self.net = nn.Sequential(
            # ── dim_hidden 폭 유지 구간 (linear1~5, 원본과 동일하게 5개) ──
            nn.Linear(dim_input, h), nn.LeakyReLU(), nn.Dropout(dropout_p),   # linear1
            nn.Linear(h, h), nn.LeakyReLU(), nn.Dropout(dropout_p),           # linear2
            nn.Linear(h, h), nn.LeakyReLU(), nn.Dropout(dropout_p),           # linear3
            nn.Linear(h, h), nn.LeakyReLU(), nn.Dropout(dropout_p),           # linear4
            nn.Linear(h, h), nn.LeakyReLU(), nn.Dropout(dropout_p),           # linear5
            # ── dim_hidden//2로 전환 후 유지 구간 (linear6~15, 원본과 동일하게 10개) ──
            nn.Linear(h, h2), nn.LeakyReLU(), nn.Dropout(dropout_p),          # linear6 (전환)
            nn.Linear(h2, h2), nn.LeakyReLU(), nn.Dropout(dropout_p),         # linear7
            nn.Linear(h2, h2), nn.LeakyReLU(), nn.Dropout(dropout_p),         # linear8
            nn.Linear(h2, h2), nn.LeakyReLU(), nn.Dropout(dropout_p),         # linear9
            nn.Linear(h2, h2), nn.LeakyReLU(), nn.Dropout(dropout_p),         # linear10
            nn.Linear(h2, h2), nn.LeakyReLU(), nn.Dropout(dropout_p),         # linear11
            nn.Linear(h2, h2), nn.LeakyReLU(), nn.Dropout(dropout_p),         # linear12
            nn.Linear(h2, h2), nn.LeakyReLU(), nn.Dropout(dropout_p),         # linear13
            nn.Linear(h2, h2), nn.LeakyReLU(), nn.Dropout(dropout_p),         # linear14
            nn.Linear(h2, h2), nn.LeakyReLU(), nn.Dropout(dropout_p),         # linear15
            # ── 출력 ──
            nn.Linear(h2, 1),                                                # linear_out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N_dest, dim_input) → logit: (N_dest,)"""
        return self.net(x).squeeze(-1)

    def multinomial_nll(self, logits: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
        """Multinomial NLL:  -sum(T * log_softmax(logit))"""
        log_p = torch.nn.functional.log_softmax(logits, dim=0)
        return -(flows * log_p).sum()
    
