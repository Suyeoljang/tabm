
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn.preprocessing
import sklearn.model_selection
import tabm
import rtdl_num_embeddings
import pickle
from typing import NamedTuple, Literal
from tqdm import tqdm
import json
from pathlib import Path
import argparse


# RegressionLabelStats를 전역으로 정의 (pickle 가능하도록)
class RegressionLabelStats(NamedTuple):
    mean: float
    std: float


# def load_best_params(study_name: str = None) -> dict:
#     """
#     Optuna 결과에서 best parameters 로드
    
#     Args:
#         study_name: Study 이름 (None이면 가장 최신 파일 사용)
    
#     Returns:
#         best parameters dict
#     """
#     results_dir = Path("optuna_results")
    
#     if study_name is None:
#         # 가장 최신 파일 찾기
#         json_files = list(results_dir.glob("*_best_params.json"))
#         if not json_files:
#             raise FileNotFoundError("optuna_results/에 best_params.json 파일이 없습니다.")
        
#         latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
#         print(f"✓ 가장 최신 파일 사용: {latest_file.name}")
#     else:
#         latest_file = results_dir / f"{study_name}_best_params.json"
#         if not latest_file.exists():
#             raise FileNotFoundError(f"{latest_file}이 존재하지 않습니다.")
    
#     with open(latest_file, 'r') as f:
#         result = json.load(f)
    
#     return result['best_params']


def main(args):
    # print("=" * 70)
    # print("최적 하이퍼파라미터로 최종 모델 학습")
    # print("=" * 70)
    
    # # ============================================================
    # # 1. 최적 하이퍼파라미터 로드
    # # ============================================================
    # print("\n1. 최적 하이퍼파라미터 로드 중...")
    
    # best_params = load_best_params(args.study_name)
    
    # print("\n최적 하이퍼파라미터:")
    # for key, value in best_params.items():
    #     print(f"  {key}: {value}")
    
    # # 파라미터 추출
    # LEARNING_RATE = best_params['learning_rate']
    # WEIGHT_DECAY = best_params['weight_decay']
    # N_BLOCKS = best_params['n_blocks']
    # D_BLOCK = best_params['d_block']
    # K = best_params['k']  # ✨ 앙상블 크기
    # N_BINS = best_params['n_bins']
    # D_EMBEDDINGS = best_params['d_embedding']
    # DROPOUT = best_params['dropout']
    # BATCH_SIZE = best_params['batch_size']

    LEARNING_RATE = 0.002
    BATCH_SIZE = 1024
    K = 32
    N_BINS = 48
    D_EMBEDDINGS = 16
    N_BLOCKS = 2
    D_BLOCK = 512
    DROPOUT = 0.1
    WEIGHT_DECAY = 3e-4
    
    # ============================================================
    # 2. 데이터 로드
    # ============================================================
    print("\n2. 데이터 로드 중...")
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    save_dir = '/mnt/user-data/outputs/'
    
    # ✨ 수정: optuna_train.py와 동일한 방식으로 로드
    df_train = pd.read_csv(f'encoded_train_data.csv')
    df_valtest = pd.read_csv(f'encoded_valtest_data.csv')
    
    split_data = np.load(f'data_split.npz')
    train_idx = split_data['train_idx']
    val_idx = split_data['val_idx']
    
    with open(f'preprocessing_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    numerical_cols = metadata['numerical_cols']
    categorical_cols = metadata['categorical_cols']
    cat_cardinalities = metadata['cat_cardinalities']
    
    print(f"✓ 데이터 로드 완료")
    print(f"  Train: {len(train_idx):,}개 (100%)")
    print(f"  Val:   {len(df_valtest):,}개 (testset과 동일)")
    print(f"  Test:  {len(df_valtest):,}개 (testset과 동일)")
    
    # ============================================================
    # 3. NumPy 배열 변환
    # ============================================================
    print("\n3. NumPy 배열 변환 중...")
    
    target_col = 'PROC_EXPOSE_LOG'
    
    # ✨ 수정: train은 전체 train 데이터, val/test는 valtest 데이터
    X_train = df_train[numerical_cols + categorical_cols].values
    y_train = df_train[target_col].values
    
    X_valtest = df_valtest[numerical_cols + categorical_cols].values
    y_valtest = df_valtest[target_col].values
    
    n_num = len(numerical_cols)
    
    data_numpy = {
        'train': {
            'x_num': X_train[:, :n_num].astype(np.float32),
            'x_cat': X_train[:, n_num:].astype(np.int64),
            'y': y_train.astype(np.float32)
        },
        'val': {
            'x_num': X_valtest[:, :n_num].astype(np.float32),
            'x_cat': X_valtest[:, n_num:].astype(np.int64),
            'y': y_valtest.astype(np.float32)
        },
        'test': {
            'x_num': X_valtest[:, :n_num].astype(np.float32),
            'x_cat': X_valtest[:, n_num:].astype(np.int64),
            'y': y_valtest.astype(np.float32)
        }
    }
    
    print(f"✓ 변환 완료")
    print(f"  Train shape: {data_numpy['train']['x_num'].shape}")
    print(f"  Val shape: {data_numpy['val']['x_num'].shape}")
    print(f"  Test shape: {data_numpy['test']['x_num'].shape}")
    
    # ============================================================
    # 4. 데이터 전처리
    # ============================================================
    print("\n4. 데이터 전처리 중...")
    
    x_num_train = data_numpy['train']['x_num']
    noise = np.random.default_rng(0).normal(0.0, 1e-5, x_num_train.shape).astype(x_num_train.dtype)
    
    preprocessing = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(len(X_train) // 30, 1000), 10),  # ✨ 수정: len(train_idx) -> len(X_train)
        output_distribution='normal',
        subsample=10**9,
    ).fit(x_num_train + noise)
    
    for part in data_numpy:
        data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])
    
    # 타겟 표준화 (Train만)
    Y_train = data_numpy['train']['y'].copy()
    regression_label_stats = RegressionLabelStats(
        Y_train.mean().item(), Y_train.std().item()
    )
    Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
    data_numpy['train']['y'] = Y_train
    
    print(f"✓ 전처리 완료")
    print(f"  타겟 mean: {regression_label_stats.mean:.6f}")
    print(f"  타겟 std:  {regression_label_stats.std:.6f}")
    
    # PyTorch 텐서 변환
    data = {
        part: {key: torch.tensor(value, device=device) 
               for key, value in part_data.items()}
        for part, part_data in data_numpy.items()
    }
    
    # ============================================================
    # 5. 모델 생성
    # ============================================================
    print("\n5. TabM 모델 생성 중...")
    
    bin_edges = rtdl_num_embeddings.compute_bins(
        torch.tensor(data_numpy['train']['x_num'], device='cpu'),
        n_bins=N_BINS
    )
    
    num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        bin_edges,
        d_embedding=D_EMBEDDINGS,
        activation=False,
        version='B',
    )
    
    model = tabm.TabM.make(
        n_num_features=len(numerical_cols),
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        dropout=DROPOUT, #DROPOUT,
        num_embeddings=num_embeddings,
        n_blocks=N_BLOCKS,
        d_block=D_BLOCK,
        k=K,  # 튜닝된 앙상블 크기
    ).to(device)
    
    print(f"✓ 모델 생성 완료")
    print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============================================================
    # 6. 학습 설정
    # ============================================================
    print("\n6. 학습 설정...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay= WEIGHT_DECAY #WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.9,
        patience=30,
    )
    
    # LogCosh Loss
    def logcosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))
    
    loss_fn = logcosh_loss
    
    print(f"학습 설정:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  K (ensemble size): {K}")
    print(f"  N_BINS: {N_BINS}")
    print(f"  D_EMBEDDINGS: {D_EMBEDDINGS}")
    print(f"  N_BLOCKS: {N_BLOCKS}")
    print(f"  D_BLOCK: {D_BLOCK}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Loss: LogCosh")
    
    # ============================================================
    # 7. 평가 함수
    # ============================================================
    @torch.no_grad()
    def evaluate(part: str) -> float:
        """모델 평가 (RMSE 반환, 음수)"""
        model.eval()
        predictions = []
        
        for batch in torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                data[part]['x_num'], 
                data[part]['x_cat']
            ),
            batch_size=1024,
            shuffle=False
        ):
            x_num_batch, x_cat_batch = batch
            pred = model(x_num_batch, x_cat_batch)
            
            # 출력 차원 확인 및 처리
            if pred.dim() == 2:
                pred = pred.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            elif pred.dim() == 3:
                pred = pred.mean(dim=1).squeeze(-1)  # [batch_size, k, 1] -> [batch_size]
            
            predictions.append(pred)
        
        predictions = torch.cat(predictions)
        
        # ✨ 수정: optuna_train.py와 동일한 방식
        # 표준화 스케일에서 RMSE 계산
        if part == 'train':
            y_true = data[part]['y']
        else:
            # Val/Test는 표준화 필요
            y_true = (data[part]['y'] - regression_label_stats.mean) / regression_label_stats.std
        
        mse = ((predictions - y_true) ** 2).mean()
        rmse = torch.sqrt(mse)
        
        return -rmse.item()  # 음수로 반환 (최대화 -> 최소화)
    
    # ============================================================
    # 8. 학습 루프
    # ============================================================
    print("\n7. 학습 시작...")
    print("=" * 70)
    
    max_epochs = 2000
    patience = 200
    best_val_score = float('-inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None
    
    # 초기 평가
    initial_val_score = evaluate('val')
    initial_test_score = evaluate('test')
    initial_val_rmse = -initial_val_score * regression_label_stats.std
    initial_test_rmse = -initial_test_score * regression_label_stats.std
    
    print(f"학습 전 Validation RMSE (원래 스케일): {initial_val_rmse:.6f}")
    print(f"학습 전 Test RMSE (원래 스케일): {initial_test_rmse:.6f}")
    print()
    
    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []
        
        # 배치 학습
        indices = torch.randperm(len(data['train']['y']), device=device)
        
        with tqdm(total=len(indices), desc=f'Epoch {epoch}', leave=False) as pbar:
            for i in range(0, len(indices), BATCH_SIZE):
                batch_indices = indices[i:i + BATCH_SIZE]
                
                x_num_batch = data['train']['x_num'][batch_indices]
                x_cat_batch = data['train']['x_cat'][batch_indices]
                y_batch = data['train']['y'][batch_indices]
                
                optimizer.zero_grad()
                predictions = model(x_num_batch, x_cat_batch)
                
                # 출력 차원 확인 및 처리
                if predictions.dim() == 2:
                    predictions = predictions.squeeze(-1)  # [batch_size, 1] -> [batch_size]
                elif predictions.dim() == 3:
                    predictions = predictions.mean(dim=1).squeeze(-1)  # [batch_size, k, 1] -> [batch_size]
                
                loss = loss_fn(predictions, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                epoch_losses.append(loss.item())
                
                pbar.update(len(batch_indices))
        
        # Epoch 평가
        mean_loss = np.mean(epoch_losses)
        val_score = evaluate('val')
        test_score = evaluate('test')
        
        # Scheduler step
        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 출력
        if epoch % 5 == 0 or val_score > best_val_score:
            marker = "*" if val_score > best_val_score else " "
            print(f"{marker} [epoch] {epoch:3d} [loss] {mean_loss:.6f} "
                  f"[val] {val_score:.6f} [test] {test_score:.6f} [lr] {current_lr:.6f}")
        
        # Best 모델 저장
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # ============================================================
    # 9. 최종 평가
    # ============================================================
    print("\n" + "=" * 70)
    print("8. 최종 평가")
    print("=" * 70)
    
    model.load_state_dict(best_state)
    
    final_val_score = evaluate('val')
    final_test_score = evaluate('test')
    
    # 원래 스케일로 복원
    final_val_rmse = -final_val_score * regression_label_stats.std
    final_test_rmse = -final_test_score * regression_label_stats.std
    
    print(f'\nBest epoch: {best_epoch}')
    print(f'표준화 스케일:')
    print(f'  Validation RMSE: {-final_val_score:.6f}')
    print(f'  Test RMSE: {-final_test_score:.6f}')
    print(f'\n원래 스케일:')
    print(f'  Validation RMSE: {final_val_rmse:.6f}')
    print(f'  Test RMSE: {final_test_rmse:.6f}')
    
    # ============================================================
    # 10. 모델 저장
    # ============================================================
    save_path = 'tabm_model_optimized.pt'
    torch.save({
        'model_state_dict': best_state,
        'regression_label_stats': regression_label_stats,
        'preprocessing': preprocessing,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'cat_cardinalities': cat_cardinalities,
        'best_epoch': best_epoch,
        'best_val_score': best_val_score,
        'final_val_rmse': final_val_rmse,
        'final_test_rmse': final_test_rmse,
        'bin_edges': bin_edges,  # 추가: inference에 필요
        'config': {
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'batch_size': BATCH_SIZE,
            'k': K,
            'n_bins': N_BINS,
            'd_embeddings': D_EMBEDDINGS,
            'dropout': DROPOUT,
            'n_blocks': N_BLOCKS,
            'd_block': D_BLOCK,
        },
        'best_params': 'None'
    }, save_path)
    
    print(f"\n✓ 모델 저장: {save_path}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with best hyperparameters')
    # parser.add_argument('--study_name', type=str, default=None,
    #                     help='tabm_tuning_20251211_104848')
    
    args = parser.parse_args()
    main(args)
