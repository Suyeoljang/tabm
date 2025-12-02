"""
TabM 모델 추론 스크립트
학습된 모델로 새로운 CSV 데이터의 PROC_EXPOSE_LOG 예측
"""

import numpy as np
import pandas as pd
import torch
import pickle
import argparse
from pathlib import Path
from typing import NamedTuple

try:
    import tabm
    import rtdl_num_embeddings
except ImportError:
    print("⚠️  tabm 패키지가 설치되지 않았습니다.")
    exit(1)


# RegressionLabelStats 정의 (모델 로드 시 필요)
class RegressionLabelStats(NamedTuple):
    mean: float
    std: float


# PyTorch 2.6 호환성: safe globals에 추가
try:
    torch.serialization.add_safe_globals([RegressionLabelStats])
except AttributeError:
    # PyTorch < 2.6에서는 이 메서드가 없음
    pass

# ================================================================
# 설정
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='TabM 모델 추론')
    parser.add_argument('--input_csv', type=str,
                        default='testset_1126_analyized.csv',
                        help='예측할 CSV 파일 경로 (예: test_data.csv)')
    parser.add_argument('--model_path', type=str, 
                        default='tabm_model_fixed.pt',
                        help='학습된 모델 파일 경로')
    parser.add_argument('--output_csv', type=str, 
                        default='predictions.csv',
                        help='예측 결과 저장 경로')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='추론 배치 크기')
    
    return parser.parse_args()


# ================================================================
# 메인 추론 함수
# ================================================================

def predict(input_csv_path, model_path, output_csv_path, batch_size=8192):
    """
    학습된 TabM 모델로 새로운 데이터 예측
    
    Args:
        input_csv_path: 예측할 CSV 파일 경로
        model_path: 학습된 모델 파일 경로
        output_csv_path: 예측 결과 저장 경로
        batch_size: 추론 배치 크기
    """
    
    print("=" * 70)
    print("TabM 모델 추론")
    print("=" * 70)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n디바이스: {device}")
    
    # ================================================================
    # 1. 모델 및 메타데이터 로드
    # ================================================================
    print("\n" + "=" * 70)
    print("1. 모델 로드")
    print("=" * 70)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 메타데이터 추출
    regression_label_stats = checkpoint['regression_label_stats']
    preprocessing = checkpoint['preprocessing']
    label_encoders = checkpoint['label_encoders']
    numerical_cols = checkpoint['numerical_cols']
    categorical_cols = checkpoint['categorical_cols']
    cat_cardinalities = checkpoint['cat_cardinalities']
    
    print(f"✓ 모델 로드: {model_path}")
    print(f"  타겟 평균: {regression_label_stats.mean:.6f}")
    print(f"  타겟 표준편차: {regression_label_stats.std:.6f}")
    print(f"  연속형 변수: {len(numerical_cols)}개")
    print(f"  범주형 변수: {len(categorical_cols)}개")
    
    # ================================================================
    # 2. 입력 데이터 로드
    # ================================================================
    print("\n" + "=" * 70)
    print("2. 입력 데이터 로드")
    print("=" * 70)
    
    df_input = pd.read_csv(input_csv_path)
    print(f"✓ 입력 데이터: {df_input.shape}")
    print(f"  파일: {input_csv_path}")
    
    # 원본 데이터 백업 (나중에 결합용)
    df_original = df_input.copy()
    
    # ================================================================
    # 3. 데이터 전처리
    # ================================================================
    print("\n" + "=" * 70)
    print("3. 데이터 전처리")
    print("=" * 70)
    
    # 3-1. 컬럼 확인
    missing_cols = []
    for col in numerical_cols + categorical_cols:
        if col not in df_input.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"⚠️  경고: 다음 컬럼이 입력 데이터에 없습니다:")
        for col in missing_cols:
            print(f"    - {col}")
        print(f"\n누락된 컬럼을 0 또는 'UNKNOWN'으로 채웁니다.")
        
        for col in missing_cols:
            if col in numerical_cols:
                df_input[col] = 0.0
            else:
                df_input[col] = 'UNKNOWN'
    
    # 3-2. 범주형 변수 인코딩
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
    
    # 3-3. numpy 배열 변환
    X_num = df_input[numerical_cols].values.astype(np.float32)
    X_cat = df_input[categorical_cols].values.astype(np.int64)
    
    # 3-4. 연속형 변수 정규화
    X_num = preprocessing.transform(X_num)
    
    print(f"✓ 전처리 완료")
    print(f"  연속형: {X_num.shape}")
    print(f"  범주형: {X_cat.shape}")
    
    # ================================================================
    # 4. 모델 생성 및 가중치 로드
    # ================================================================
    print("\n" + "=" * 70)
    print("4. 모델 생성")
    print("=" * 70)
    
    n_num_features = len(numerical_cols)
    n_cat_features = len(categorical_cols)
    
    # 추론용 num_embeddings 생성
    # ⚠️ 중요: 학습 시 사용한 n_bins=48을 그대로 사용해야 함!
    n_samples = len(X_num)
    
    # bins 계산을 위한 랜덤 데이터 생성 (1000개)
    # compute_bins는 내부적으로 n_bins = min(n_bins, len(X)-1)로 조정하므로
    # n_bins=48을 보장하려면 최소 49개, 안전하게 1000개 사용
    np.random.seed(42)
    X_num_for_bins = np.random.randn(1000, n_num_features).astype(np.float32)
    
    # 실제 데이터의 범위에 맞춰 스케일 조정
    if n_samples > 0:
        data_min = X_num.min(axis=0)
        data_max = X_num.max(axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0  # 0 division 방지
        
        # 랜덤 데이터를 실제 데이터 범위로 스케일
        X_num_for_bins = X_num_for_bins * data_range + data_min
    
    n_bins = 16  # 항상 48로 고정 (학습 시와 동일)
    
    print(f"  샘플 수: {n_samples}")
    print(f"  n_bins: {n_bins} (학습 시와 동일하게 고정)")
    
    # Num embeddings 생성
    num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(
            torch.tensor(X_num_for_bins, device='cpu'),
            n_bins=n_bins
        ),
        d_embedding=16,
        activation=False,
        version='B',
    )
    
    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        num_embeddings=num_embeddings,
    ).to(device)
    
    # 가중치 로드 (strict=False로 mask 등 불필요한 키 무시)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✓ 모델 생성 완료")
    print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # ================================================================
    # 5. 예측
    # ================================================================
    print("\n" + "=" * 70)
    print("5. 예측 중...")
    print("=" * 70)
    
    # PyTorch 텐서 변환
    X_num_tensor = torch.tensor(X_num, device=device)
    X_cat_tensor = torch.tensor(X_cat, device=device)
    
    predictions_list = []
    
    with torch.no_grad():
        # 배치 단위로 예측
        n_samples = len(X_num)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            
            x_num_batch = X_num_tensor[i:batch_end]
            x_cat_batch = X_cat_tensor[i:batch_end]
            
            # 예측
            y_pred_batch = model(x_num_batch, x_cat_batch)
            
            # (batch_size, k, 1) → (batch_size, k) → (batch_size,)
            y_pred_batch = y_pred_batch.squeeze(-1).mean(dim=1)
            
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
    # 6. 결과 저장
    # ================================================================
    print("\n" + "=" * 70)
    print("6. 결과 저장")
    print("=" * 70)
    
    # 원본 데이터에 예측 결과 추가
    df_result = df_original.copy()
    df_result['PROC_EXPOSE_LOG_PRED'] = predictions
    
    # CSV 저장
    df_result.to_csv(output_csv_path, index=False)
    
    print(f"✓ 결과 저장: {output_csv_path}")
    print(f"  컬럼: {list(df_result.columns)}")
    print(f"  행 수: {len(df_result)}")
    
    # ================================================================
    # 7. 요약 통계
    # ================================================================
    print("\n" + "=" * 70)
    print("7. 예측 요약")
    print("=" * 70)
    
    print(f"\n예측값 통계:")
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
        print("8. 성능 평가 (실제값 존재)")
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
        args.batch_size
    )
    
    print(f"\n✅ 성공!")
    print(f"예측 결과: {args.output_csv}")
