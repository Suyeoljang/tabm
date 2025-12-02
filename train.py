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


# (tabm) suyeol@node4:~/project/tabm$ python train_new.py
# ======================================================================
# TabM 학습 - 타겟 표준화 버그 수정 버전
# ======================================================================
# 디바이스: cuda:0

# ======================================================================
# 1. 데이터 로드
# ======================================================================
# ✓ 데이터 로드 완료
#   타겟: PROC_EXPOSE_LOG
#   Train: 374,622개
#   Val:   41,625개
#   Test:  52,905개 (별도 파일)

# ======================================================================
# 2. numpy 배열 변환
# ======================================================================
# ✓ 변환 완료
#   Train: x_num (374622, 6), x_cat (374622, 14)
#   Val:   x_num (41625, 6), x_cat (41625, 14)
#   Test:  x_num (52905, 6), x_cat (52905, 14)

# ======================================================================
# 3. 데이터 전처리
# ======================================================================
# ✓ 전처리 완료
#   타겟 mean: 4.533395
#   타겟 std:  1.055436

# ======================================================================
# 데이터 준비 완료! 이제 학습을 시작할 수 있습니다.
# ======================================================================

# ======================================================================
# 4. TabM 모델 생성
# ======================================================================
# ✓ 모델 생성 완료
#   파라미터: 3,166,240

# ======================================================================
# 5. 학습 설정
# ======================================================================
# 학습 설정:
#   Learning rate: 0.002
#   Weight decay: 0.004
#   Batch size: 512
#   Max epochs: 1000
#   N_BINS: 16
#   D_EMBEDDINGS: 16
#   Dropout: 0.1
#   Loss: LogCosh
#   N_BLOCKS: 4
#   D_BLOCK: 512

# ======================================================================
# 6. 학습 시작
# ======================================================================

# 학습 전 Test RMSE (표준화 스케일): 1.048242
# * [epoch] 0   [loss] 0.007780 [val] -0.036331 [test] -0.038164 [lr] 0.002000
# * [epoch] 1   [loss] 0.001545 [val] -0.034097 [test] -0.035342 [lr] 0.002000
# * [epoch] 2   [loss] 0.001213 [val] -0.029999 [test] -0.032384 [lr] 0.002000
# * [epoch] 4   [loss] 0.000985 [val] -0.026932 [test] -0.029514 [lr] 0.002000
# * [epoch] 5   [loss] 0.000934 [val] -0.026758 [test] -0.029006 [lr] 0.002000
# * [epoch] 6   [loss] 0.000884 [val] -0.026486 [test] -0.028985 [lr] 0.002000
# * [epoch] 7   [loss] 0.000855 [val] -0.026338 [test] -0.029181 [lr] 0.002000
# * [epoch] 8   [loss] 0.000827 [val] -0.025751 [test] -0.028575 [lr] 0.002000
# * [epoch] 9   [loss] 0.000818 [val] -0.025439 [test] -0.028349 [lr] 0.002000
# * [epoch] 10  [loss] 0.000799 [val] -0.025048 [test] -0.027594 [lr] 0.002000
# * [epoch] 13  [loss] 0.000749 [val] -0.024665 [test] -0.028040 [lr] 0.002000
# * [epoch] 15  [loss] 0.000727 [val] -0.024252 [test] -0.027377 [lr] 0.002000
# * [epoch] 18  [loss] 0.000706 [val] -0.023958 [test] -0.027346 [lr] 0.002000
#   [epoch] 20  [loss] 0.000688 [val] -0.026266 [test] -0.029550 [lr] 0.002000
#   [epoch] 25  [loss] 0.000660 [val] -0.024832 [test] -0.028697 [lr] 0.002000
# * [epoch] 26  [loss] 0.000657 [val] -0.023456 [test] -0.027209 [lr] 0.002000
# * [epoch] 29  [loss] 0.000646 [val] -0.023229 [test] -0.027145 [lr] 0.002000
#   [epoch] 30  [loss] 0.000640 [val] -0.024330 [test] -0.027891 [lr] 0.002000
#   [epoch] 35  [loss] 0.000628 [val] -0.023610 [test] -0.027381 [lr] 0.002000
# * [epoch] 38  [loss] 0.000619 [val] -0.023129 [test] -0.027329 [lr] 0.002000
#   [epoch] 40  [loss] 0.000616 [val] -0.023764 [test] -0.027331 [lr] 0.002000
# * [epoch] 42  [loss] 0.000610 [val] -0.023013 [test] -0.026986 [lr] 0.002000
#   [epoch] 45  [loss] 0.000607 [val] -0.023264 [test] -0.027284 [lr] 0.002000
#   [epoch] 50  [loss] 0.000599 [val] -0.024279 [test] -0.028287 [lr] 0.002000
#   [epoch] 55  [loss] 0.000592 [val] -0.023435 [test] -0.027586 [lr] 0.002000
# * [epoch] 58  [loss] 0.000588 [val] -0.022691 [test] -0.027029 [lr] 0.002000
#   [epoch] 60  [loss] 0.000586 [val] -0.023603 [test] -0.027206 [lr] 0.002000
#   [epoch] 65  [loss] 0.000581 [val] -0.023258 [test] -0.027442 [lr] 0.002000
#   [epoch] 70  [loss] 0.000576 [val] -0.023455 [test] -0.027721 [lr] 0.002000
#   [epoch] 75  [loss] 0.000575 [val] -0.023730 [test] -0.028093 [lr] 0.002000
#   [epoch] 80  [loss] 0.000571 [val] -0.023835 [test] -0.027805 [lr] 0.002000
#   [epoch] 85  [loss] 0.000568 [val] -0.024172 [test] -0.028335 [lr] 0.002000
#   [epoch] 90  [loss] 0.000564 [val] -0.022987 [test] -0.027468 [lr] 0.002000
#   [epoch] 95  [loss] 0.000562 [val] -0.023358 [test] -0.027634 [lr] 0.002000
#   [epoch] 100 [loss] 0.000555 [val] -0.022995 [test] -0.027272 [lr] 0.001800
#   [epoch] 105 [loss] 0.000554 [val] -0.022696 [test] -0.027209 [lr] 0.001800
#   [epoch] 110 [loss] 0.000553 [val] -0.023210 [test] -0.027589 [lr] 0.001800
#   [epoch] 115 [loss] 0.000552 [val] -0.022786 [test] -0.027265 [lr] 0.001800
#   [epoch] 120 [loss] 0.000549 [val] -0.023070 [test] -0.027379 [lr] 0.001800
#   [epoch] 125 [loss] 0.000549 [val] -0.023056 [test] -0.027275 [lr] 0.001800
#   [epoch] 130 [loss] 0.000548 [val] -0.022961 [test] -0.027591 [lr] 0.001800
#   [epoch] 135 [loss] 0.000546 [val] -0.023399 [test] -0.027932 [lr] 0.001800
#   [epoch] 140 [loss] 0.000546 [val] -0.023382 [test] -0.027550 [lr] 0.001620
#   [epoch] 145 [loss] 0.000538 [val] -0.023003 [test] -0.027183 [lr] 0.001620
#   [epoch] 150 [loss] 0.000540 [val] -0.023050 [test] -0.027357 [lr] 0.001620
#   [epoch] 155 [loss] 0.000538 [val] -0.023881 [test] -0.028493 [lr] 0.001620
#   [epoch] 160 [loss] 0.000539 [val] -0.022914 [test] -0.027602 [lr] 0.001620
#   [epoch] 165 [loss] 0.000537 [val] -0.024023 [test] -0.028038 [lr] 0.001620
# * [epoch] 166 [loss] 0.000536 [val] -0.022596 [test] -0.027046 [lr] 0.001620
# * [epoch] 167 [loss] 0.000536 [val] -0.022338 [test] -0.026853 [lr] 0.001620
#   [epoch] 170 [loss] 0.000537 [val] -0.022918 [test] -0.027493 [lr] 0.001620
#   [epoch] 175 [loss] 0.000534 [val] -0.023166 [test] -0.027818 [lr] 0.001620
#   [epoch] 180 [loss] 0.000534 [val] -0.023030 [test] -0.027575 [lr] 0.001620
# * [epoch] 185 [loss] 0.000533 [val] -0.022330 [test] -0.026875 [lr] 0.001620
#   [epoch] 190 [loss] 0.000533 [val] -0.023100 [test] -0.027467 [lr] 0.001620
#   [epoch] 195 [loss] 0.000532 [val] -0.022970 [test] -0.027639 [lr] 0.001620
#   [epoch] 200 [loss] 0.000532 [val] -0.022883 [test] -0.027446 [lr] 0.001620
#   [epoch] 205 [loss] 0.000532 [val] -0.022683 [test] -0.027320 [lr] 0.001620
#   [epoch] 210 [loss] 0.000531 [val] -0.022620 [test] -0.027041 [lr] 0.001620
#   [epoch] 215 [loss] 0.000532 [val] -0.022704 [test] -0.027386 [lr] 0.001620
#   [epoch] 220 [loss] 0.000530 [val] -0.023386 [test] -0.027556 [lr] 0.001620
#   [epoch] 225 [loss] 0.000531 [val] -0.023218 [test] -0.027950 [lr] 0.001620
#   [epoch] 230 [loss] 0.000526 [val] -0.023346 [test] -0.027902 [lr] 0.001458
#   [epoch] 235 [loss] 0.000526 [val] -0.023221 [test] -0.027575 [lr] 0.001458
#   [epoch] 240 [loss] 0.000525 [val] -0.023588 [test] -0.028182 [lr] 0.001458
#   [epoch] 245 [loss] 0.000526 [val] -0.022998 [test] -0.027635 [lr] 0.001458
#   [epoch] 250 [loss] 0.000525 [val] -0.023509 [test] -0.028240 [lr] 0.001458
#   [epoch] 255 [loss] 0.000523 [val] -0.022535 [test] -0.027124 [lr] 0.001458
#   [epoch] 260 [loss] 0.000524 [val] -0.022747 [test] -0.027340 [lr] 0.001458
#   [epoch] 265 [loss] 0.000523 [val] -0.023891 [test] -0.028629 [lr] 0.001458
#   [epoch] 270 [loss] 0.000520 [val] -0.023036 [test] -0.027499 [lr] 0.001312
#   [epoch] 275 [loss] 0.000519 [val] -0.023654 [test] -0.028407 [lr] 0.001312
#   [epoch] 280 [loss] 0.000520 [val] -0.023367 [test] -0.027856 [lr] 0.001312
#   [epoch] 285 [loss] 0.000520 [val] -0.022512 [test] -0.027206 [lr] 0.001312
#   [epoch] 290 [loss] 0.000519 [val] -0.022663 [test] -0.027473 [lr] 0.001312
#   [epoch] 295 [loss] 0.000518 [val] -0.022960 [test] -0.027573 [lr] 0.001312
#   [epoch] 300 [loss] 0.000518 [val] -0.023843 [test] -0.028631 [lr] 0.001312
#   [epoch] 305 [loss] 0.000519 [val] -0.023440 [test] -0.028184 [lr] 0.001312
#   [epoch] 310 [loss] 0.000515 [val] -0.022907 [test] -0.027686 [lr] 0.001181
#   [epoch] 315 [loss] 0.000515 [val] -0.022494 [test] -0.027159 [lr] 0.001181
# * [epoch] 318 [loss] 0.000515 [val] -0.022199 [test] -0.026838 [lr] 0.001181
#   [epoch] 320 [loss] 0.000514 [val] -0.023348 [test] -0.028323 [lr] 0.001181
#   [epoch] 325 [loss] 0.000514 [val] -0.022698 [test] -0.027429 [lr] 0.001181
#   [epoch] 330 [loss] 0.000514 [val] -0.023138 [test] -0.027563 [lr] 0.001181
#   [epoch] 335 [loss] 0.000514 [val] -0.022734 [test] -0.027580 [lr] 0.001181
#   [epoch] 340 [loss] 0.000514 [val] -0.023488 [test] -0.028367 [lr] 0.001181
#   [epoch] 345 [loss] 0.000514 [val] -0.023260 [test] -0.027853 [lr] 0.001181
#   [epoch] 350 [loss] 0.000513 [val] -0.023339 [test] -0.028140 [lr] 0.001181
#   [epoch] 355 [loss] 0.000514 [val] -0.022796 [test] -0.027654 [lr] 0.001181
#   [epoch] 360 [loss] 0.000511 [val] -0.023088 [test] -0.027863 [lr] 0.001063
#   [epoch] 365 [loss] 0.000511 [val] -0.022951 [test] -0.027729 [lr] 0.001063
#   [epoch] 370 [loss] 0.000510 [val] -0.023384 [test] -0.028346 [lr] 0.001063
#   [epoch] 375 [loss] 0.000511 [val] -0.022698 [test] -0.027425 [lr] 0.001063
#   [epoch] 380 [loss] 0.000510 [val] -0.022974 [test] -0.027923 [lr] 0.001063
#   [epoch] 385 [loss] 0.000510 [val] -0.022997 [test] -0.027693 [lr] 0.001063
#   [epoch] 390 [loss] 0.000510 [val] -0.022581 [test] -0.027169 [lr] 0.001063
#   [epoch] 395 [loss] 0.000510 [val] -0.022694 [test] -0.027639 [lr] 0.001063
#   [epoch] 400 [loss] 0.000509 [val] -0.022890 [test] -0.027551 [lr] 0.000957
#   [epoch] 405 [loss] 0.000507 [val] -0.023025 [test] -0.027825 [lr] 0.000957
#   [epoch] 410 [loss] 0.000507 [val] -0.022739 [test] -0.027591 [lr] 0.000957
#   [epoch] 415 [loss] 0.000507 [val] -0.022839 [test] -0.027634 [lr] 0.000957
#   [epoch] 420 [loss] 0.000506 [val] -0.022932 [test] -0.027845 [lr] 0.000957
#   [epoch] 425 [loss] 0.000506 [val] -0.022976 [test] -0.027614 [lr] 0.000957
#   [epoch] 430 [loss] 0.000506 [val] -0.022740 [test] -0.027556 [lr] 0.000957
#   [epoch] 435 [loss] 0.000506 [val] -0.022959 [test] -0.027730 [lr] 0.000957
#   [epoch] 440 [loss] 0.000507 [val] -0.022870 [test] -0.027811 [lr] 0.000957
#   [epoch] 445 [loss] 0.000504 [val] -0.022693 [test] -0.027663 [lr] 0.000861
#   [epoch] 450 [loss] 0.000504 [val] -0.022567 [test] -0.027493 [lr] 0.000861
#   [epoch] 455 [loss] 0.000504 [val] -0.022710 [test] -0.027611 [lr] 0.000861
#   [epoch] 460 [loss] 0.000504 [val] -0.022393 [test] -0.027195 [lr] 0.000861
#   [epoch] 465 [loss] 0.000503 [val] -0.022868 [test] -0.027733 [lr] 0.000861
#   [epoch] 470 [loss] 0.000504 [val] -0.023293 [test] -0.027848 [lr] 0.000861
#   [epoch] 475 [loss] 0.000504 [val] -0.022830 [test] -0.027571 [lr] 0.000861
#   [epoch] 480 [loss] 0.000504 [val] -0.023135 [test] -0.028004 [lr] 0.000861
#   [epoch] 485 [loss] 0.000501 [val] -0.022572 [test] -0.027382 [lr] 0.000775
#   [epoch] 490 [loss] 0.000501 [val] -0.023041 [test] -0.027921 [lr] 0.000775
#   [epoch] 495 [loss] 0.000501 [val] -0.022497 [test] -0.027460 [lr] 0.000775
#   [epoch] 500 [loss] 0.000501 [val] -0.022652 [test] -0.027639 [lr] 0.000775
#   [epoch] 505 [loss] 0.000501 [val] -0.022863 [test] -0.027622 [lr] 0.000775
#   [epoch] 510 [loss] 0.000501 [val] -0.022733 [test] -0.027668 [lr] 0.000775
#   [epoch] 515 [loss] 0.000501 [val] -0.022388 [test] -0.027215 [lr] 0.000775

# Early stopping at epoch 518

# ======================================================================
# 7. 최종 평가
# ======================================================================

# Best epoch: 318
# 표준화 스케일:
#   Validation RMSE: 0.022199
#   Test RMSE: 0.026838

# 원래 스케일:
#   Validation RMSE: 0.023430
#   Test RMSE: 0.028326

# ✓ 모델 저장: tabm_model_fixed.pt
# ======================================================================
