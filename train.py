"""
TabM 학습 - 타겟 표준화 버그 수정 버전
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

class LogCoshLoss(nn.Module):
    """Log-Cosh Loss: 매끄러운 Huber"""
    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        loss = torch.log(torch.cosh(error))
        return loss.mean()

# ================================================================
# 설정
# ================================================================
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 4e-3
BATCH_SIZE = 512
N_EPOCHS = 1000
PATIENCE = 200
N_BINS = 32
D_EMBEDDINGS = 32
DROPOUT = 0.1
N_BLOCKS = 4  # 현재 4에서 증가
D_BLOCK = 256

print("=" * 70)
print("TabM 학습 - 타겟 표준화 버그 수정 버전")
print("=" * 70)

# 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)

# 디바이스 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"디바이스: {device}")

# ================================================================
# 1. 데이터 로드 (수정된 부분)
# ================================================================
print("\n" + "=" * 70)
print("1. 데이터 로드")
print("=" * 70)

data_dir = '/mnt/user-data/outputs/'

# Train/Val 데이터 로드
df_trainval = pd.read_csv(f'encoded_trainval_data.csv')

# Test 데이터 로드 (별도 파일)
df_test = pd.read_csv(f'encoded_test_data.csv')

# 분할 인덱스 로드
split_data = np.load(f'data_split.npz')
train_idx = split_data['train_idx']
val_idx = split_data['val_idx']
test_size = split_data['test_size']

# 메타데이터 로드
with open(f'preprocessing_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

label_encoders = metadata['label_encoders']
cat_cardinalities = metadata['cat_cardinalities']
numerical_cols = metadata['numerical_cols']
categorical_cols = metadata['categorical_cols']
target_col = metadata['target_col']

print(f"✓ 데이터 로드 완료")
print(f"  타겟: {target_col}")
print(f"  Train: {len(train_idx):,}개")
print(f"  Val:   {len(val_idx):,}개")
print(f"  Test:  {len(df_test):,}개 (별도 파일)")

# Train/Val 데이터 분리
y_trainval = df_trainval[target_col].values
X_trainval = df_trainval.drop(columns=[target_col])

# Test 데이터
y_test = df_test[target_col].values
X_test = df_test.drop(columns=[target_col])

# ================================================================
# 2. numpy 배열 변환 (수정된 부분)
# ================================================================
print("\n" + "=" * 70)
print("2. numpy 배열 변환")
print("=" * 70)

# Train/Val에서 numerical, categorical 분리
X_num_trainval = X_trainval[numerical_cols].values
X_cat_trainval = X_trainval[categorical_cols].values
Y_numpy_trainval = y_trainval

# Test에서 numerical, categorical 분리
X_num_test = X_test[numerical_cols].values
X_cat_test = X_test[categorical_cols].values
Y_numpy_test = y_test

data_numpy = {
    'train': {
        'x_num': X_num_trainval[train_idx].astype(np.float32),
        'x_cat': X_cat_trainval[train_idx].astype(np.int64),
        'y': Y_numpy_trainval[train_idx].astype(np.float32)
    },
    'val': {
        'x_num': X_num_trainval[val_idx].astype(np.float32),
        'x_cat': X_cat_trainval[val_idx].astype(np.int64),
        'y': Y_numpy_trainval[val_idx].astype(np.float32)
    },
    'test': {
        'x_num': X_num_test.astype(np.float32),
        'x_cat': X_cat_test.astype(np.int64),
        'y': Y_numpy_test.astype(np.float32)
    }
}

print("✓ 변환 완료")
print(f"  Train: x_num {data_numpy['train']['x_num'].shape}, x_cat {data_numpy['train']['x_cat'].shape}")
print(f"  Val:   x_num {data_numpy['val']['x_num'].shape}, x_cat {data_numpy['val']['x_cat'].shape}")
print(f"  Test:  x_num {data_numpy['test']['x_num'].shape}, x_cat {data_numpy['test']['x_cat'].shape}")

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

print("\n" + "=" * 70)
print("데이터 준비 완료! 이제 학습을 시작할 수 있습니다.")
print("=" * 70)


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
        n_bins=N_BINS
    ),
    d_embedding=D_EMBEDDINGS,
    activation=False,
    version='B',
)

model = tabm.TabM.make(
    n_num_features=n_num_features,
    cat_cardinalities=cat_cardinalities,
    d_out=1,
    dropout=DROPOUT,
    num_embeddings=num_embeddings,
    n_blocks = N_BLOCKS,  # 현재 4에서 증가
    d_block = D_BLOCK
).to(device)

print(f"✓ 모델 생성 완료")
print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}")

# ================================================================
# 6. 학습 설정
# ================================================================
print("\n" + "=" * 70)
print("5. 학습 설정")
print("=" * 70)

optimizer = torch.optim.AdamW(model.parameters(), 
                               lr=LEARNING_RATE, 
                               weight_decay=WEIGHT_DECAY)
gradient_clipping_norm: Optional[float] = 1.0
share_training_batches = True

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', 
                              factor=0.9, patience=35, 
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

criterion = LogCoshLoss()

print(f"학습 설정:")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {N_EPOCHS}")
print(f"  N_BINS: {N_BINS}")
print(f"  D_EMBEDDINGS: {D_EMBEDDINGS}")
print(f"  Dropout: {DROPOUT}")
print(f"  Loss: LogCosh")
print(f"  N_BLOCKS: {N_BLOCKS}")
print(f"  D_BLOCK: {D_BLOCK}")

# ================================================================
# 7. 학습 함수
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
    """표준화된 스케일에서 평가 (역변환 안 함!)"""
    model.eval()
    
    y_pred: np.ndarray = (
        torch.cat([
            apply_model(part, idx)
            for idx in torch.arange(len(data[part]['y']), device=device).split(eval_batch_size)
        ])
        .cpu()
        .numpy()
    )
    
    # Ensemble mean (k개 모델 평균)
    y_pred = y_pred*regression_label_stats.std+regression_label_stats.mean
    y_pred = y_pred.mean(1)
    
    # 표준화된 스케일 그대로 사용!
    y_true = data[part]['y'].cpu().numpy()
    
    # RMSE 계산 (표준화된 스케일에서)
    score = -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
    
    return float(score)

# ================================================================
# 8. 학습 시작
# ================================================================
print("\n" + "=" * 70)
print("6. 학습 시작")
print("=" * 70)

print(f'\n학습 전 Test RMSE (표준화 스케일): {-evaluate("test"):.6f}')

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
# 9. 최종 평가
# ================================================================
print("\n" + "=" * 70)
print("7. 최종 평가")
print("=" * 70)

model.load_state_dict(best_state)

final_val_score = evaluate('val')
final_test_score = evaluate('test')

# 원래 스케일로 복원해서 출력
final_val_rmse = -final_val_score * regression_label_stats.std
final_test_rmse = -final_test_score * regression_label_stats.std

print(f'\nBest epoch: {best_epoch}')
print(f'표준화 스케일:')
print(f'  Validation RMSE: {-final_val_score:.6f}')
print(f'  Test RMSE: {-final_test_score:.6f}')
print(f'\n원래 스케일:')
print(f'  Validation RMSE: {final_val_rmse:.6f}')
print(f'  Test RMSE: {final_test_rmse:.6f}')

# ================================================================
# 10. 모델 저장
# ================================================================
save_path = 'tabm_model_fixed.pt'
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
    'config': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
        'n_bins': N_BINS,
        'd_embeddings': D_EMBEDDINGS,
        'dropout': DROPOUT,
    }
}, save_path)

print(f"\n✓ 모델 저장: {save_path}")
print("=" * 70)
