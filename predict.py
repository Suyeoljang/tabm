#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import pickle
import argparse
from pathlib import Path
from typing import Tuple, NamedTuple
import json
import tabm
import rtdl_num_embeddings


# RegressionLabelStats를 전역으로 정의
class RegressionLabelStats(NamedTuple):
    """회귀 타겟의 평균과 표준편차를 저장하는 클래스"""
    mean: float
    std: float


def load_best_params(study_name: str = None) -> dict:
    """
    Optuna 결과에서 best parameters 로드
    
    Args:
        study_name: Study 이름 (None이면 가장 최신 파일 사용)
    
    Returns:
        best parameters dict
    """
    results_dir = Path("optuna_results")
    
    if study_name is None:
        # 가장 최신 파일 찾기
        json_files = list(results_dir.glob("*_best_params.json"))
        if not json_files:
            raise FileNotFoundError("optuna_results/에 best_params.json 파일이 없습니다.")
        
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"✓ 가장 최신 파일 사용: {latest_file.name}")
    else:
        latest_file = results_dir / f"{study_name}_best_params.json"
        if not latest_file.exists():
            raise FileNotFoundError(f"{latest_file}이 존재하지 않습니다.")
    
    with open(latest_file, 'r') as f:
        result = json.load(f)
    
    return result['best_params']


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='TabM 모델 추론')
    parser.add_argument('--input_csv', type=str, default='testset_1126_add.csv',
                        help='입력 CSV 파일 경로')
    parser.add_argument('--model_path', type=str, default='tabm_model_optimized.pt',
                        help='학습된 모델 체크포인트 경로')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                        help='예측 결과 저장 경로')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='추론 배치 크기')
    parser.add_argument('--study_name', type=str, default='tabm_tuning_20251211_104848',
                        help='Optuna study name')
    return parser.parse_args()


def predict(
    input_csv_path: str,
    model_path: str,
    output_csv_path: str = 'predictions.csv',
    batch_size: int = 8192,
    study_name: str = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    TabM 모델로 예측 수행
    
    Args:
        input_csv_path: 입력 CSV 파일 경로
        model_path: 학습된 모델 체크포인트 경로
        output_csv_path: 결과 저장 경로
        batch_size: 배치 크기
        study_name: Optuna study name
        
    Returns:
        predictions: 예측값 배열
        df_result: 결과 데이터프레임
    """
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 70)
    print("TabM 모델 추론")
    print("=" * 70)
    print(f"\n디바이스: {device}")
    
    # ================================================================
    # 1. Best parameters 로드
    # ================================================================
    print("\n" + "=" * 70)
    print("1. 하이퍼파라미터 로드")
    print("=" * 70)
    
    best_params = load_best_params(study_name)
    
    LEARNING_RATE = best_params['learning_rate']
    WEIGHT_DECAY = best_params['weight_decay']
    N_BLOCKS = best_params['n_blocks']
    D_BLOCK = best_params['d_block']
    K = best_params['k']
    N_BINS = best_params['n_bins']
    D_EMBEDDINGS = best_params['d_embedding']
    DROPOUT = best_params['dropout']
    BATCH_SIZE = best_params['batch_size']
    
    print(f"✓ 하이퍼파라미터 로드 완료")
    print(f"  k: {K}")
    print(f"  n_blocks: {N_BLOCKS}")
    print(f"  d_block: {D_BLOCK}")
    print(f"  n_bins: {N_BINS}")
    
    # ================================================================
    # 2. 모델 체크포인트 로드
    # ================================================================
    print("\n" + "=" * 70)
    print("2. 모델 체크포인트 로드")
    print("=" * 70)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 메타데이터 추출
    regression_label_stats = checkpoint['regression_label_stats']
    preprocessing = checkpoint['preprocessing']
    
    # label_encoders가 체크포인트에 없으면 별도 파일에서 로드
    if 'label_encoders' in checkpoint:
        label_encoders = checkpoint['label_encoders']
        numerical_cols = checkpoint['numerical_cols']
        categorical_cols = checkpoint['categorical_cols']
        cat_cardinalities = checkpoint['cat_cardinalities']
        print("✓ 메타데이터를 체크포인트에서 로드")
    else:
        print("\n⚠️  체크포인트에 label_encoders가 없습니다.")
        print("   preprocessing_metadata.pkl에서 로드합니다...")
        
        with open('preprocessing_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        label_encoders = metadata['label_encoders']
        numerical_cols = metadata['numerical_cols']
        categorical_cols = metadata['categorical_cols']
        cat_cardinalities = metadata['cat_cardinalities']
        
        print("   ✓ 메타데이터 로드 완료!")
    
    config = checkpoint.get('config', {})
    
    # bin_edges 로드
    if 'bin_edges' not in checkpoint:
        print("\n⚠️  경고: 체크포인트에 'bin_edges'가 없습니다!")
        print("   임시로 랜덤 데이터로 bins를 생성합니다.")
        
        n_num_features = len(numerical_cols)
        np.random.seed(42)
        X_num_for_bins = np.random.randn(1000, n_num_features).astype(np.float32)
        
        bin_edges = rtdl_num_embeddings.compute_bins(
            torch.tensor(X_num_for_bins, device='cpu'),
            n_bins=N_BINS
        )
    else:
        bin_edges = checkpoint['bin_edges']
        print(f"✓ Bins 로드 완료!")
    
    print(f"\n✓ 모델 로드: {model_path}")
    print(f"  타겟 평균: {regression_label_stats.mean:.6f}")
    print(f"  타겟 표준편차: {regression_label_stats.std:.6f}")
    print(f"  연속형 변수: {len(numerical_cols)}개")
    print(f"  범주형 변수: {len(categorical_cols)}개")
    
    # ================================================================
    # 3. 입력 데이터 로드
    # ================================================================
    print("\n" + "=" * 70)
    print("3. 입력 데이터 로드")
    print("=" * 70)
    
    df_input = pd.read_csv(input_csv_path)
    print(f"✓ 입력 데이터: {df_input.shape}")
    print(f"  파일: {input_csv_path}")
    
    # 원본 데이터 백업
    df_original = df_input.copy()
    
    # ================================================================
    # 4. 데이터 전처리
    # ================================================================
    print("\n" + "=" * 70)
    print("4. 데이터 전처리")
    print("=" * 70)
    
    # 4-1. 컬럼 확인 및 추가
    missing_cols = []
    for col in numerical_cols + categorical_cols:
        if col not in df_input.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"⚠️  경고: 다음 컬럼이 입력 데이터에 없습니다:")
        for col in missing_cols:
            print(f"    - {col}")
        print(f"\n누락된 컬럼을 기본값으로 채웁니다.")
        
        for col in missing_cols:
            if col in numerical_cols:
                df_input[col] = 0.0
            else:
                df_input[col] = 'UNKNOWN'
    
    # 4-2. 범주형 변수 인코딩
    print("\n범주형 변수 인코딩 중...")
    
    for col in categorical_cols:
        le = label_encoders[col]
        
        def safe_transform(x):
            try:
                return le.transform([str(x)])[0]
            except ValueError:
                # 학습 시 보지 못한 카테고리 → UNKNOWN
                return len(le.classes_)  # UNKNOWN 인덱스
        
        df_input[col] = df_input[col].astype(str).apply(safe_transform)
    
    print(f"✓ 범주형 인코딩 완료")
    
    # 4-3. numpy 배열 변환
    X_num = df_input[numerical_cols].values.astype(np.float32)
    X_cat = df_input[categorical_cols].values.astype(np.int64)
    
    # 4-4. 연속형 변수 정규화
    X_num = preprocessing.transform(X_num)
    
    print(f"✓ 전처리 완료")
    print(f"  연속형: {X_num.shape}")
    print(f"  범주형: {X_cat.shape}")
    
    # ================================================================
    # 5. 모델 생성 및 가중치 로드
    # ================================================================
    print("\n" + "=" * 70)
    print("5. 모델 생성")
    print("=" * 70)
    
    n_num_features = len(numerical_cols)
    
    num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        bin_edges,
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
        n_blocks=N_BLOCKS,
        d_block=D_BLOCK,
        k=K,
    ).to(device)
    
    # 가중치 로드 (strict=False로 mask 등 불필요한 키 무시)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✓ 모델 생성 완료")
    print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # ================================================================
    # 6. 예측
    # ================================================================
    print("\n" + "=" * 70)
    print("6. 예측 중...")
    print("=" * 70)
    
    # PyTorch 텐서 변환
    X_num_tensor = torch.tensor(X_num, device=device)
    X_cat_tensor = torch.tensor(X_cat, device=device)
    
    predictions_list = []
    
    with torch.no_grad():
        n_samples = len(X_num)
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            
            x_num_batch = X_num_tensor[i:batch_end]
            x_cat_batch = X_cat_tensor[i:batch_end]
            
            # 예측
            y_pred_batch = model(x_num_batch, x_cat_batch)
            
            # 출력 차원 처리
            if y_pred_batch.dim() == 2:
                y_pred_batch = y_pred_batch.squeeze(-1)
            elif y_pred_batch.dim() == 3:
                # (batch_size, k, 1) → (batch_size,)
                y_pred_batch = y_pred_batch.mean(dim=1).squeeze(-1)
            
            predictions_list.append(y_pred_batch.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0 or batch_end == n_samples:
                print(f"  진행: {batch_end}/{n_samples} ({batch_end/n_samples*100:.1f}%)")
    
    # 전체 예측 결합
    predictions = np.concatenate(predictions_list)
    
    # 역표준화 (원래 스케일로)
    predictions = predictions * regression_label_stats.std + regression_label_stats.mean
    
    print(f"\n✓ 예측 완료!")
    print(f"  예측값 범위: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"  예측값 평균: {predictions.mean():.6f}")
    print(f"  예측값 표준편차: {predictions.std():.6f}")
    
    # ================================================================
    # 7. 결과 저장
    # ================================================================
    print("\n" + "=" * 70)
    print("7. 결과 저장")
    print("=" * 70)
    
    # 원본 데이터에 예측 결과 추가
    df_result = df_original.copy()
    df_result['PROC_EXPOSE_LOG_PRED'] = predictions
    df_result['PRED_PROC_EXPOSE'] = np.expm1(predictions)
    
    # CSV 저장
    df_result.to_csv(output_csv_path, index=False)
    
    print(f"✓ 결과 저장: {output_csv_path}")
    print(f"  컬럼 수: {len(df_result.columns)}개")
    print(f"  행 수: {len(df_result)}개")
    
    # ================================================================
    # 8. 요약 통계
    # ================================================================
    print("\n" + "=" * 70)
    print("8. 예측 요약")
    print("=" * 70)
    
    print(f"\n예측값 통계 (PROC_EXPOSE_LOG):")
    print(f"  최솟값:   {predictions.min():.6f}")
    print(f"  최댓값:   {predictions.max():.6f}")
    print(f"  평균:     {predictions.mean():.6f}")
    print(f"  중앙값:   {np.median(predictions):.6f}")
    print(f"  표준편차: {predictions.std():.6f}")
    
    # 백분위수
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n백분위수:")
    for p in percentiles:
        val = np.percentile(predictions, p)
        print(f"  {p:2d}%: {val:.6f}")
    
    # 실제값이 있는지 확인
    if 'PROC_EXPOSE_LOG' in df_original.columns:
        print("\n" + "=" * 70)
        print("9. 성능 평가 (실제값 존재)")
        print("=" * 70)
        
        y_true = df_original['PROC_EXPOSE_LOG'].values
        y_pred = predictions
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # 상대 오차
        rel_error = rmse / y_true.mean() * 100
        
        print(f"\n평가 지표:")
        print(f"  RMSE:        {rmse:.6f}")
        print(f"  MAE:         {mae:.6f}")
        print(f"  R²:          {r2:.6f}")
        print(f"  상대 오차:   {rel_error:.2f}%")
        
        # 결과에 오차도 추가
        df_result['PROC_EXPOSE_LOG_TRUE'] = y_true
        df_result['ERROR'] = y_true - y_pred
        df_result['ABS_ERROR'] = np.abs(y_true - y_pred)
        df_result.to_csv(output_csv_path, index=False)
        
        print(f"\n✓ 오차 정보도 결과에 추가됨 (ERROR, ABS_ERROR)")
    
    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)
    
    return predictions, df_result


# ================================================================
# 메인 실행
# ================================================================

if __name__ == "__main__":
    args = parse_args()
    
    print("\n설정:")
    print(f"  입력 CSV:   {args.input_csv}")
    print(f"  모델 경로:  {args.model_path}")
    print(f"  출력 CSV:   {args.output_csv}")
    print(f"  배치 크기:  {args.batch_size}")
    print(f"  Study name: {args.study_name}")
    
    # 파일 존재 확인
    if not Path(args.input_csv).exists():
        print(f"\n❌ 오류: 입력 파일이 없습니다: {args.input_csv}")
        exit(1)
    
    if not Path(args.model_path).exists():
        print(f"\n❌ 오류: 모델 파일이 없습니다: {args.model_path}")
        exit(1)
    
    # 예측 실행
    predictions, df_result = predict(
        args.input_csv,
        args.model_path,
        args.output_csv,
        args.batch_size,
        args.study_name
    )
    
    print(f"\n✅ 성공!")
    print(f"예측 결과: {args.output_csv}")
