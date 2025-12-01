"""
TabM 모델 추론 스크립트 (Target Encoding 지원)
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
    pass

# ================================================================
# 설정
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='TabM 모델 추론')
    parser.add_argument('--input_csv', type=str,
                        default='testset_1126_analyized.csv',
                        help='예측할 CSV 파일 경로')
    parser.add_argument('--model_path', type=str, 
                        default='tabm_model_fixed.pt',
                        help='학습된 모델 파일 경로')
    parser.add_argument('--preprocessing_meta', type=str,
                        default='preprocessing_metadata.pkl',
                        help='전처리 메타데이터 파일 경로')
    parser.add_argument('--output_csv', type=str, 
                        default='predictions.csv',
                        help='예측 결과 저장 경로')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='추론 배치 크기')
    
    return parser.parse_args()


# ================================================================
# 인코딩 함수 추가
# ================================================================

def apply_frequency_encoding(df, col, freq_info):
    """Frequency Encoding 적용"""
    freq_dict = freq_info['freq_dict']
    return df[col].map(freq_dict).fillna(0).values

def apply_target_encoding(df, col, target_info):
    """Target Encoding 적용 (inference)"""
    mean_dict = target_info['mean_dict']
    global_mean = target_info['global_mean']
    return df[col].map(mean_dict).fillna(global_mean).values


# ================================================================
# 메인 추론 함수
# ================================================================

def predict(input_csv_path, model_path, preprocessing_meta_path, output_csv_path, batch_size=8192):
    """
    학습된 TabM 모델로 새로운 데이터 예측
    
    Args:
        input_csv_path: 예측할 CSV 파일 경로
        model_path: 학습된 모델 파일 경로
        preprocessing_meta_path: 전처리 메타데이터 경로
        output_csv_path: 예측 결과 저장 경로
        batch_size: 추론 배치 크기
    """
    
    print("=" * 70)
    print("TabM 모델 추론 (최적 인코딩 전략)")
    print("=" * 70)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n디바이스: {device}")
    
    # ================================================================
    # 1. 모델 및 메타데이터 로드
    # ================================================================
    print("\n" + "=" * 70)
    print("1. 모델 및 메타데이터 로드")
    print("=" * 70)
    
    # 모델 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    regression_label_stats = checkpoint['regression_label_stats']
    preprocessing = checkpoint['preprocessing']
    label_encoders = checkpoint['label_encoders']
    numerical_cols = checkpoint['numerical_cols']
    categorical_cols = checkpoint['categorical_cols']
    cat_cardinalities = checkpoint['cat_cardinalities']
    config = checkpoint.get('config', {})
    
    # 전처리 메타데이터 로드 (새로운 인코딩 전략)
    with open(preprocessing_meta_path, 'rb') as f:
        preprocessing_meta = pickle.load(f)
    
    encoding_info = preprocessing_meta.get('encoding_info', {})
    encoding_strategy = preprocessing_meta.get('encoding_strategy', {})
    
    # 원본 범주형 컬럼 리스트 구성
    # original_categorical_cols가 메타데이터에 있으면 사용, 없으면 직접 구성
    if 'original_categorical_cols' in preprocessing_meta:
        original_categorical_cols = preprocessing_meta['original_categorical_cols']
    else:
        # categorical_cols + very_high_card_cols
        original_categorical_cols = (
            categorical_cols + 
            encoding_strategy.get('very_high_card', [])
        )
    
    very_high_card_cols = encoding_strategy.get('very_high_card', [])
    high_card_cols = encoding_strategy.get('high_card', [])
    medium_card_cols = encoding_strategy.get('medium_card', [])
    low_card_cols = encoding_strategy.get('low_card', [])
    
    print(f"✓ 모델 로드: {model_path}")
    print(f"  타겟 평균: {regression_label_stats.mean:.6f}")
    print(f"  타겟 표준편차: {regression_label_stats.std:.6f}")
    print(f"  연속형 변수: {len(numerical_cols)}개")
    print(f"  범주형 변수: {len(categorical_cols)}개")
    
    print(f"\n  인코딩 전략:")
    print(f"    Very High Card (>300):  {len(very_high_card_cols)}개")
    print(f"    High Card (151-300):    {len(high_card_cols)}개")
    print(f"    Medium Card (51-150):   {len(medium_card_cols)}개")
    print(f"    Low Card (≤50):         {len(low_card_cols)}개")
    
    # ================================================================
    # 2. 입력 데이터 로드
    # ================================================================
    print("\n" + "=" * 70)
    print("2. 입력 데이터 로드")
    print("=" * 70)
    
    df_input = pd.read_csv(input_csv_path)
    print(f"✓ 입력 데이터: {df_input.shape}")
    print(f"  파일: {input_csv_path}")
    
    # 원본 데이터 백업
    df_original = df_input.copy()
    
    # ================================================================
    # 필수: 학습에 사용된 원본 컬럼만 추출
    # ================================================================
    # 원본 수치형 컬럼 (인코딩 전)
    original_numerical_cols = [col for col in numerical_cols 
                              if not any(col.endswith(suffix) for suffix in ['_target', '_freq'])]
    
    # 필요한 모든 원본 컬럼
    required_original_cols = list(set(original_numerical_cols + original_categorical_cols))
    
    # 누락된 컬럼 확인
    missing_cols = [col for col in required_original_cols if col not in df_input.columns]
    
    if missing_cols:
        print(f"\n⚠️  경고: 다음 컬럼이 입력 데이터에 없습니다:")
        for col in missing_cols:
            print(f"    - {col}")
        print(f"\n누락된 컬럼을 기본값으로 채웁니다.")
        
        for col in missing_cols:
            if col in original_numerical_cols:
                df_input[col] = 0.0
            else:
                df_input[col] = 'UNKNOWN'
    
    # 필요한 컬럼만 선택 (불필요한 컬럼 제거)
    extra_cols = [col for col in df_input.columns if col not in required_original_cols]
    if extra_cols:
        print(f"\n📌 입력 데이터에서 {len(extra_cols)}개의 불필요한 컬럼 제거")
        print(f"   필요한 컬럼: {len(required_original_cols)}개")
        df_input = df_input[required_original_cols].copy()
    
    print(f"\n✓ 원본 컬럼 추출 완료")
    print(f"  수치형: {len(original_numerical_cols)}개")
    print(f"  범주형: {len(original_categorical_cols)}개")
    
    # ================================================================
    # 3. 데이터 전처리 (새로운 인코딩 전략 적용)
    # ================================================================
    print("\n" + "=" * 70)
    print("3. 데이터 전처리")
    print("=" * 70)
    
    # 3-1. Very High Cardinality (>300): K-Fold Target Encoding만
    if very_high_card_cols:
        print(f"\nVery High Cardinality 변수 처리 (K-Fold Target Encoding):")
        
        for col in very_high_card_cols:
            if col not in df_input.columns:
                print(f"  ⚠️  경고: {col} 컬럼이 입력 데이터에 없습니다.")
                continue
                
            if col in encoding_info and encoding_info[col]['type'] == 'kfold_target':
                info = encoding_info[col]
                
                # K-Fold Target Encoding 적용
                target_encoded = apply_target_encoding(df_input, col, info['target_info'])
                new_col = f'{col}_target'
                df_input[new_col] = target_encoded
                
                print(f"  ✓ {col} → {new_col}")
                print(f"    학습된 카테고리: {len(info['target_info']['mean_dict'])}개")
                
                # 원본 컬럼 제거
                df_input = df_input.drop(columns=[col])
    
    # 3-2. High Cardinality (151-300): Label + Frequency + Target
    if high_card_cols:
        print(f"\nHigh Cardinality 변수 처리 (Label + Freq + Target):")
        
        for col in high_card_cols:
            if col not in df_input.columns:
                print(f"  ⚠️  경고: {col} 컬럼이 입력 데이터에 없습니다.")
                continue
                
            if col in encoding_info and encoding_info[col]['type'] == 'high_card':
                info = encoding_info[col]
                
                # Frequency Encoding
                freq_encoded = apply_frequency_encoding(df_input, col, info['freq_info'])
                df_input[f'{col}_freq'] = freq_encoded
                
                # Target Encoding
                target_encoded = apply_target_encoding(df_input, col, info['target_info'])
                df_input[f'{col}_target'] = target_encoded
                
                print(f"  ✓ {col} → _freq, _target")
    
    # 3-3. Medium Cardinality (51-150): Label + Frequency
    if medium_card_cols:
        print(f"\nMedium Cardinality 변수 처리 (Label + Freq):")
        
        for col in medium_card_cols:
            if col not in df_input.columns:
                print(f"  ⚠️  경고: {col} 컬럼이 입력 데이터에 없습니다.")
                continue
                
            if col in encoding_info and encoding_info[col]['type'] == 'medium_card':
                info = encoding_info[col]
                
                # Frequency Encoding
                freq_encoded = apply_frequency_encoding(df_input, col, info['freq_info'])
                df_input[f'{col}_freq'] = freq_encoded
                
                print(f"  ✓ {col} → _freq")
    
    # 3-4. Low Cardinality: Label Only (별도 처리 없음)
    if low_card_cols:
        print(f"\nLow Cardinality 변수: Label Encoding만 적용 ({len(low_card_cols)}개)")
    
    # 3-5. 범주형 변수 Label Encoding
    print(f"\n범주형 변수 Label Encoding:")
    
    for col in categorical_cols:
        if col not in df_input.columns:
            print(f"  ⚠️  경고: {col} 컬럼이 입력 데이터에 없습니다. 'UNKNOWN'으로 설정")
            df_input[col] = 'UNKNOWN'
            
        if col in label_encoders:
            le = label_encoders[col]
            
            def safe_transform(x):
                try:
                    return le.transform([str(x)])[0]
                except ValueError:
                    return len(le.classes_)  # UNKNOWN
            
            df_input[col] = df_input[col].astype(str).apply(safe_transform)
    
    print(f"✓ 범주형 인코딩 완료")
    
    # 3-6. 누락된 연속형 컬럼 확인 및 채우기
    for col in numerical_cols:
        if col not in df_input.columns:
            print(f"  ⚠️  경고: {col} 컬럼이 입력 데이터에 없습니다. 0으로 채웁니다.")
            df_input[col] = 0.0
    
    # 3-7. numpy 배열 변환
    X_num = df_input[numerical_cols].values.astype(np.float32)
    X_cat = df_input[categorical_cols].values.astype(np.int64)
    
    # 3-8. 연속형 변수 정규화
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
    
    # config에서 n_bins, d_embeddings 가져오기
    n_bins = config.get('n_bins', 24)
    d_embeddings = config.get('d_embeddings', 32)
    dropout = config.get('dropout', 0.1)
    
    print(f"  n_bins: {n_bins}")
    print(f"  d_embeddings: {d_embeddings}")
    print(f"  dropout: {dropout}")
    
    # bins 계산을 위한 랜덤 데이터 생성
    np.random.seed(42)
    X_num_for_bins = np.random.randn(1000, n_num_features).astype(np.float32)
    
    # 실제 데이터의 범위에 맞춰 스케일 조정
    n_samples = len(X_num)
    if n_samples > 0:
        data_min = X_num.min(axis=0)
        data_max = X_num.max(axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0
        X_num_for_bins = X_num_for_bins * data_range + data_min
    
    # Num embeddings 생성
    num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(
            torch.tensor(X_num_for_bins, device='cpu'),
            n_bins=n_bins
        ),
        d_embedding=d_embeddings,
        activation=False,
        version='B',
    )
    
    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        num_embeddings=num_embeddings,
        dropout=dropout,
    ).to(device)
    
    # 가중치 로드
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
        n_samples = len(X_num)
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            
            x_num_batch = X_num_tensor[i:batch_end]
            x_cat_batch = X_cat_tensor[i:batch_end]
            
            # 예측
            y_pred_batch = model(x_num_batch, x_cat_batch)
            
            # (batch_size, k, 1) → (batch_size,)
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
    
    # exp 변환 (로그였다면)
    df_result['PROC_EXPOSE_PRED'] = np.expm1(predictions)
    
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
    print(f"  전처리 메타: {args.preprocessing_meta}")
    print(f"  출력 CSV:   {args.output_csv}")
    print(f"  배치 크기:  {args.batch_size}")
    
    # 파일 존재 확인
    if not Path(args.input_csv).exists():
        print(f"\n❌ 오류: 입력 파일이 없습니다: {args.input_csv}")
        exit(1)
    
    if not Path(args.model_path).exists():
        print(f"\n❌ 오류: 모델 파일이 없습니다: {args.model_path}")
        exit(1)
        
    if not Path(args.preprocessing_meta).exists():
        print(f"\n❌ 오류: 전처리 메타데이터 파일이 없습니다: {args.preprocessing_meta}")
        exit(1)
    
    # 예측 실행
    predictions, df_result = predict(
        args.input_csv,
        args.model_path,
        args.preprocessing_meta,
        args.output_csv,
        args.batch_size
    )
    
    print(f"\n✅ 성공!")
    print(f"예측 결과: {args.output_csv}")
