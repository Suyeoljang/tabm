"""
TabM 학습 - 다양한 Loss Function 지원
PROC_EXPOSE_LOG 예측에 적합한 loss 찾기
"""

import math
import random
from typing import Literal, NamedTuple, Optional
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.special
import sklearn.metrics
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor
import pickle

# TabM 관련 임포트
try:
    import tabm
    import rtdl_num_embeddings
except ImportError:
    print("⚠️  tabm 패키지가 설치되지 않았습니다.")
    exit(1)

# ================================================================
# 커스텀 Loss Functions
# ================================================================

class HuberLoss(nn.Module):
    """Huber Loss: MSE + MAE의 장점 결합"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()


class LogCoshLoss(nn.Module):
    """Log-Cosh Loss: 매끄러운 Huber"""
    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        loss = torch.log(torch.cosh(error))
        return loss.mean()


class MSLELoss(nn.Module):
    """Mean Squared Log Error: 상대 오차 중시"""
    def forward(self, y_pred, y_true):
        # log(1+x)를 사용하여 0 값 처리
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        loss = (log_pred - log_true) ** 2
        return loss.mean()


class QuantileLoss(nn.Module):
    """Quantile Loss: 특정 백분위수 예측"""
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile
    
    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        return loss.mean()


# ================================================================
# 설정
# ================================================================
print("=" * 70)
print("TabM 학습 - 다양한 Loss Function")
print("=" * 70)

# ================================================================
# LOSS FUNCTION 선택 (여기서 변경!)
# ================================================================
LOSS_TYPE = 'huber'  # 'mse', 'mae', 'huber', 'logcosh', 'msle', 'quantile'
HUBER_DELTA = 1.0    # Huber loss의 delta 파라미터
QUANTILE = 0.5       # Quantile loss의 quantile 파라미터

print(f"\n선택된 Loss: {LOSS_TYPE.upper()}")
if LOSS_TYPE == 'huber':
    print(f"  Huber delta: {HUBER_DELTA}")
elif LOSS_TYPE == 'quantile':
    print(f"  Quantile: {QUANTILE}")

# 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"디바이스: {device}")

# ================================================================
# 1. 전처리된 데이터 로드
# ================================================================
print("\n" + "=" * 70)
print("1. 데이터 로드")
print("=" * 70)

data_dir = '/mnt/user-data/outputs/'

df = pd.read_csv('encoded_data.csv')
split_data = np.load('data_split.npz')
train_idx = split_data['train_idx']
val_idx = split_data['val_idx']
test_idx = split_data['test_idx']

with open('preprocessing_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

label_encoders = metadata['label_encoders']
cat_cardinalities = metadata['cat_cardinalities']
numerical_cols = metadata['numerical_cols']
categorical_cols = metadata['categorical_cols']
target_col = metadata['target_col']

print(f"✓ 데이터 로드 완료")
print(f"  타겟: {target_col}")
print(f"  Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

y = df[target_col].values
X = df.drop(columns=[target_col])

# ================================================================
# 2. numpy 배열 변환
# ================================================================
print("\n" + "=" * 70)
print("2. numpy 배열 변환")
print("=" * 70)

X_num = X[numerical_cols].values
X_cat = X[categorical_cols].values
Y_numpy = y

data_numpy = {
    'train': {'x_num': X_num[train_idx].astype(np.float32),
              'x_cat': X_cat[train_idx].astype(np.int64),
              'y': Y_numpy[train_idx].astype(np.float32)},
    'val': {'x_num': X_num[val_idx].astype(np.float32),
            'x_cat': X_cat[val_idx].astype(np.int64),
            'y': Y_numpy[val_idx].astype(np.float32)},
    'test': {'x_num': X_num[test_idx].astype(np.float32),
             'x_cat': X_cat[test_idx].astype(np.int64),
             'y': Y_numpy[test_idx].astype(np.float32)}
}

print("✓ 변환 완료")

# ================================================================
# 3. 데이터 전처리
# ================================================================
print("\n" + "=" * 70)
print("3. 데이터 전처리")
print("=" * 70)

x_num_train = data_numpy['train']['x_num']
noise = np.random.default_rng(0).normal(0.0, 1e-5, x_num_train.shape).astype(x_num_train.dtype)

preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution='normal',
    subsample=10**9,
).fit(x_num_train + noise)

for part in data_numpy:
    data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])

# 타겟 표준화
class RegressionLabelStats(NamedTuple):
    mean: float
    std: float

task_type: Literal['regression', 'binclass', 'multiclass'] = 'regression'
n_classes = None

Y_train = data_numpy['train']['y'].copy()
regression_label_stats = RegressionLabelStats(
    Y_train.mean().item(), Y_train.std().item()
)
Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
data_numpy['train']['y'] = Y_train

print(f"✓ 전처리 완료")
print(f"  타겟 mean: {regression_label_stats.mean:.6f}")
print(f"  타겟 std:  {regression_label_stats.std:.6f}")

# ================================================================
# 4. PyTorch 텐서 변환
# ================================================================
data = {
    part: {key: torch.tensor(value, device=device) 
           for key, value in part_data.items()}
    for part, part_data in data_numpy.items()
}

# ================================================================
# 5. TabM 모델 생성
# ================================================================
print("\n" + "=" * 70)
print("4. TabM 모델 생성")
print("=" * 70)

n_num_features = len(numerical_cols)
n_cat_features = len(categorical_cols)

num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
    rtdl_num_embeddings.compute_bins(
        torch.tensor(data_numpy['train']['x_num'], device='cpu'),
        n_bins=48
    ),
    d_embedding=32,
    activation=False,
    version='B',
)

model = tabm.TabM.make(
    n_num_features=n_num_features,
    cat_cardinalities=cat_cardinalities,
    d_out=1,
    num_embeddings=num_embeddings,
).to(device)

print(f"✓ 모델 생성 완료")
print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}")

# ================================================================
# 6. 학습 설정
# ================================================================
print("\n" + "=" * 70)
print("5. 학습 설정")
print("=" * 70)

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 256
N_EPOCHS = 300
PATIENCE = 20

optimizer = torch.optim.AdamW(model.parameters(), 
                               lr=LEARNING_RATE, 
                               weight_decay=WEIGHT_DECAY)
gradient_clipping_norm: Optional[float] = 1.0
share_training_batches = True

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', 
                              factor=0.8, patience=20, 
                              min_lr=1e-7)

if device.type == 'cuda':
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory_gb >= 40:
        eval_batch_size = 65536
    elif gpu_memory_gb >= 20:
        eval_batch_size = 32768
    else:
        eval_batch_size = 16384
else:
    eval_batch_size = 4096

print(f"✓ 학습 설정:")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {N_EPOCHS}")
print(f"  Loss function: {LOSS_TYPE.upper()}")

# ================================================================
# 7. Loss Function 설정
# ================================================================

# Loss function 선택
if LOSS_TYPE == 'mse':
    criterion = nn.MSELoss()
elif LOSS_TYPE == 'mae':
    criterion = nn.L1Loss()
elif LOSS_TYPE == 'huber':
    criterion = HuberLoss(delta=HUBER_DELTA)
elif LOSS_TYPE == 'logcosh':
    criterion = LogCoshLoss()
elif LOSS_TYPE == 'msle':
    criterion = MSLELoss()
elif LOSS_TYPE == 'quantile':
    criterion = QuantileLoss(quantile=QUANTILE)
else:
    raise ValueError(f"Unknown loss type: {LOSS_TYPE}")

print(f"  Loss object: {criterion}")

# ================================================================
# 8. 학습 함수
# ================================================================

def apply_model(part: str, idx: Tensor) -> Tensor:
    x_num = data[part]['x_num'][idx]
    x_cat = data[part]['x_cat'][idx]
    return model(x_num, x_cat)


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_pred = y_pred.flatten(0, 1).squeeze(-1)
    
    if share_training_batches:
        y_true = y_true.repeat_interleave(model.backbone.k)
    else:
        y_true = y_true.flatten(0, 1)
    
    return criterion(y_pred, y_true)


@torch.no_grad()
def evaluate(part: str) -> float:
    model.eval()
    
    y_pred: np.ndarray = (
        torch.cat([
            apply_model(part, idx)
            for idx in torch.arange(len(data[part]['y']), device=device).split(eval_batch_size)
        ])
        .cpu()
        .numpy()
    )
    
    y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean
    y_pred = y_pred.mean(1)
    
    y_true = data[part]['y'].cpu().numpy()
    
    # RMSE 계산 (평가 지표는 항상 RMSE)
    score = -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
    
    return float(score)

# ================================================================
# 9. 학습 시작
# ================================================================
print("\n" + "=" * 70)
print("6. 학습 시작")
print("=" * 70)

print(f'\n학습 전 Test RMSE: {-evaluate("test"):.6f}')

best_val_score = -np.inf
best_epoch = -1
best_state = None
no_improvement_count = 0

for epoch in range(N_EPOCHS):
    model.train()
    
    epoch_losses = []
    for batch_idx in torch.randperm(len(data['train']['y']), device=device).split(BATCH_SIZE):
        optimizer.zero_grad()
        
        y_pred = apply_model('train', batch_idx)
        
        if share_training_batches:
            y_true = data['train']['y'][batch_idx]
        else:
            batch_idx_k = torch.stack([
                torch.randperm(len(data['train']['y']), device=device)[:len(batch_idx)]
                for _ in range(model.backbone.k)
            ], dim=1)
            y_true = data['train']['y'][batch_idx_k]
        
        loss = loss_fn(y_pred, y_true)
        epoch_losses.append(loss.item())
        
        loss.backward()
        
        if gradient_clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)
        
        optimizer.step()
    
    avg_loss = np.mean(epoch_losses)
    val_score = evaluate('val')
    test_score = evaluate('test')
    
    scheduler.step(val_score)
    current_lr = optimizer.param_groups[0]['lr']
    
    if val_score > best_val_score:
        best_val_score = val_score
        best_epoch = epoch
        best_state = deepcopy(model.state_dict())
        no_improvement_count = 0
        print(f'* [epoch] {epoch:<3} [loss] {avg_loss:.6f} [val] {val_score:.6f} [test] {test_score:.6f} [lr] {current_lr:.6f}')
    else:
        no_improvement_count += 1
        if epoch % 5 == 0:
            print(f'  [epoch] {epoch:<3} [loss] {avg_loss:.6f} [val] {val_score:.6f} [test] {test_score:.6f} [lr] {current_lr:.6f}')
    
    if no_improvement_count >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch}')
        break

# ================================================================
# 10. 최종 평가
# ================================================================
print("\n" + "=" * 70)
print("7. 최종 평가")
print("=" * 70)

model.load_state_dict(best_state)

final_val_score = evaluate('val')
final_test_score = evaluate('test')

print(f'\nBest epoch: {best_epoch}')
print(f'Loss function: {LOSS_TYPE.upper()}')
print(f'Final Validation RMSE: {-final_val_score:.6f}')
print(f'Final Test RMSE: {-final_test_score:.6f}')

# ================================================================
# 11. 모델 저장
# ================================================================
save_path = f'tabm_model_{LOSS_TYPE}.pt'
torch.save({
    'model_state_dict': best_state,
    'regression_label_stats': regression_label_stats,
    'preprocessing': preprocessing,
    'label_encoders': label_encoders,
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'cat_cardinalities': cat_cardinalities,
    'best_epoch': best_epoch,
    'best_val_score': best_val_score,
    'loss_type': LOSS_TYPE,
    'config': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
    }
}, save_path)

print(f"\n✓ 모델 저장: {save_path}")
print("=" * 70)
