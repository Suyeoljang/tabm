import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import sklearn.model_selection

# ================================================================
# 1. 데이터 로드
# ================================================================
print("=" * 70)
print("데이터 전처리 및 저장 (최적 인코딩 전략)")
print("=" * 70)

# 1번 CSV: Train/Validation용
df_trainval = pd.read_csv('cd_trainset.csv')
print(f"\n1. 데이터 로드")
print(f"   Train/Val 파일: {df_trainval.shape}")

# 2번 CSV: Test용
df_test = pd.read_csv('cd_testset.csv')
print(f"   Test 파일: {df_test.shape}")

# 타겟 분리
target_col = 'PROC_EXPOSE_LOG'

# Train/Val 데이터
y_trainval = df_trainval[target_col].values
X_trainval = df_trainval.drop(columns=[target_col])

# Test 데이터
y_test = df_test[target_col].values
X_test = df_test.drop(columns=[target_col])

# Feature 타입 분리 (Train/Val 기준)
numerical_cols = X_trainval.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_trainval.select_dtypes(include=['object']).columns.tolist()

print(f"   연속형 변수: {len(numerical_cols)}개")
print(f"   범주형 변수: {len(categorical_cols)}개")

# ================================================================
# 2. Train/Val 분할 (90% / 10%)
# ================================================================
print("\n2. 데이터 분할 (Train 90% / Val 10%)")

seed = 42
train_idx, val_idx = sklearn.model_selection.train_test_split(
    np.arange(len(X_trainval)), test_size=0.1, random_state=seed
)

print(f"   Train: {len(train_idx)} samples ({len(train_idx)/len(X_trainval)*100:.1f}%)")
print(f"   Val:   {len(val_idx)} samples ({len(val_idx)/len(X_trainval)*100:.1f}%)")
print(f"   Test:  {len(X_test)} samples (별도 파일)")

# ================================================================
# 3. 변수별 카디널리티 분석 및 인코딩 전략 결정
# ================================================================
print("\n3. 변수별 인코딩 전략 결정")

# 카디널리티 임계값 설정
VERY_LOW_CARD_THRESHOLD = 10      # 10개 이하: Label Encoding
LOW_CARD_THRESHOLD = 50            # 50개 이하: Label Encoding
MEDIUM_CARD_THRESHOLD = 150        # 150개 이하: Label + Frequency Encoding
HIGH_CARD_THRESHOLD = 300          # 300개 초과: K-Fold Target Encoding

# 변수별 카디널리티 계산 및 분류
very_low_card_cols = []    # <= 10: Label만
low_card_cols = []         # 11-50: Label만
medium_card_cols = []      # 51-150: Label + Frequency
high_card_cols = []        # 151-300: Label + Frequency + Target
very_high_card_cols = []   # > 300: K-Fold Target Encoding만

for col in categorical_cols:
    n_unique = X_trainval.iloc[train_idx][col].nunique()
    
    if n_unique <= VERY_LOW_CARD_THRESHOLD:
        very_low_card_cols.append(col)
        strategy = "Label Only"
    elif n_unique <= LOW_CARD_THRESHOLD:
        low_card_cols.append(col)
        strategy = "Label Only"
    elif n_unique <= MEDIUM_CARD_THRESHOLD:
        medium_card_cols.append(col)
        strategy = "Label + Frequency"
    elif n_unique <= HIGH_CARD_THRESHOLD:
        high_card_cols.append(col)
        strategy = "Label + Frequency + Target"
    else:
        very_high_card_cols.append(col)
        strategy = "K-Fold Target (Primary)"
    
    print(f"   {col:20s}: {n_unique:4d}개 → {strategy}")

print(f"\n인코딩 전략 요약:")
print(f"   Very Low (≤10):     {len(very_low_card_cols)}개 - Label만")
print(f"   Low (11-50):        {len(low_card_cols)}개 - Label만")
print(f"   Medium (51-150):    {len(medium_card_cols)}개 - Label + Frequency")
print(f"   High (151-300):     {len(high_card_cols)}개 - Label + Freq + Target")
print(f"   Very High (>300):   {len(very_high_card_cols)}개 - K-Fold Target (Primary)")

# ================================================================
# 4. K-Fold Target Encoding 함수
# ================================================================
def kfold_target_encode(X_train, y_train, X_val, X_test, col, n_folds=5, smoothing=10):
    """
    K-Fold Target Encoding (Data Leakage 방지)
    
    Args:
        smoothing: 평활화 파라미터 (작은 샘플 수에 대한 regularization)
    """
    # 1. Train set: K-Fold로 out-of-fold encoding
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_encoded = np.zeros(len(X_train))
    
    for fold_idx, (fit_idx, transform_idx) in enumerate(kf.split(X_train)):
        # fit_idx로 평균 계산
        fit_df = pd.DataFrame({
            'category': X_train.iloc[fit_idx][col],
            'target': y_train[fit_idx]
        })
        
        # 카테고리별 통계
        stats = fit_df.groupby('category')['target'].agg(['mean', 'count'])
        global_mean = y_train[fit_idx].mean()
        
        # Smoothing 적용 (small sample regularization)
        stats['smoothed_mean'] = (
            (stats['count'] * stats['mean'] + smoothing * global_mean) /
            (stats['count'] + smoothing)
        )
        
        mean_dict = stats['smoothed_mean'].to_dict()
        
        # transform_idx에 적용
        train_encoded[transform_idx] = (
            X_train.iloc[transform_idx][col]
            .map(mean_dict)
            .fillna(global_mean)
        )
    
    # 2. Validation/Test: 전체 Train으로 계산
    train_df = pd.DataFrame({
        'category': X_train[col],
        'target': y_train
    })
    
    stats_full = train_df.groupby('category')['target'].agg(['mean', 'count'])
    global_mean_full = y_train.mean()
    
    # Smoothing 적용
    stats_full['smoothed_mean'] = (
        (stats_full['count'] * stats_full['mean'] + smoothing * global_mean_full) /
        (stats_full['count'] + smoothing)
    )
    
    mean_dict_full = stats_full['smoothed_mean'].to_dict()
    
    val_encoded = X_val[col].map(mean_dict_full).fillna(global_mean_full).values
    test_encoded = X_test[col].map(mean_dict_full).fillna(global_mean_full).values
    
    # 인코딩 정보 저장
    encoding_info = {
        'mean_dict': mean_dict_full,
        'global_mean': global_mean_full,
        'n_folds': n_folds,
        'smoothing': smoothing
    }
    
    return train_encoded, val_encoded, test_encoded, encoding_info

# ================================================================
# 5. Frequency Encoding 함수
# ================================================================
def frequency_encode(X_train, X_val, X_test, col):
    """
    Frequency Encoding: 카테고리별 출현 빈도로 인코딩
    """
    # Train에서 빈도 계산
    freq_dict = X_train[col].value_counts(normalize=True).to_dict()
    
    # 적용
    train_encoded = X_train[col].map(freq_dict).fillna(0).values
    val_encoded = X_val[col].map(freq_dict).fillna(0).values
    test_encoded = X_test[col].map(freq_dict).fillna(0).values
    
    encoding_info = {
        'freq_dict': freq_dict
    }
    
    return train_encoded, val_encoded, test_encoded, encoding_info

# ================================================================
# 6. Train/Val/Test 데이터 준비
# ================================================================
X_train = X_trainval.iloc[train_idx].copy()
y_train = y_trainval[train_idx]
X_val = X_trainval.iloc[val_idx].copy()
X_test_copy = X_test.copy()

# 인코딩 정보 저장용
all_encoding_info = {}
new_numerical_cols = numerical_cols.copy()

# ================================================================
# 7. Very High Cardinality (>300): K-Fold Target Encoding (Primary)
# ================================================================
if very_high_card_cols:
    print(f"\n4. Very High Cardinality 변수 처리 (K-Fold Target Encoding)")
    
    for col in very_high_card_cols:
        print(f"   처리 중: {col} (K-Fold Target Encoding)")
        
        # K-Fold Target Encoding
        train_enc, val_enc, test_enc, enc_info = kfold_target_encode(
            X_train, y_train, X_val, X_test_copy, col, n_folds=5, smoothing=10
        )
        
        new_col = f'{col}_target'
        
        # 적용
        X_trainval.loc[train_idx, new_col] = train_enc
        X_trainval.loc[val_idx, new_col] = val_enc
        X_test[new_col] = test_enc
        
        new_numerical_cols.append(new_col)
        all_encoding_info[col] = {
            'type': 'kfold_target',
            'target_info': enc_info
        }
        
        # 원본 컬럼 제거
        X_trainval = X_trainval.drop(columns=[col])
        X_test = X_test.drop(columns=[col])
        categorical_cols.remove(col)

# ================================================================
# 8. High Cardinality (151-300): Label + Frequency + Target
# ================================================================
if high_card_cols:
    print(f"\n5. High Cardinality 변수 처리 (Label + Frequency + Target)")
    
    for col in high_card_cols:
        print(f"   처리 중: {col}")
        
        # (1) Frequency Encoding
        train_freq, val_freq, test_freq, freq_info = frequency_encode(
            X_train, X_val, X_test_copy, col
        )
        
        freq_col = f'{col}_freq'
        X_trainval.loc[train_idx, freq_col] = train_freq
        X_trainval.loc[val_idx, freq_col] = val_freq
        X_test[freq_col] = test_freq
        new_numerical_cols.append(freq_col)
        
        # (2) K-Fold Target Encoding (smoothing 더 강하게)
        train_tgt, val_tgt, test_tgt, tgt_info = kfold_target_encode(
            X_train, y_train, X_val, X_test_copy, col, n_folds=5, smoothing=20
        )
        
        tgt_col = f'{col}_target'
        X_trainval.loc[train_idx, tgt_col] = train_tgt
        X_trainval.loc[val_idx, tgt_col] = val_tgt
        X_test[tgt_col] = test_tgt
        new_numerical_cols.append(tgt_col)
        
        # (3) Label Encoding (TabM의 embedding을 위해 유지)
        all_encoding_info[col] = {
            'type': 'high_card',
            'freq_info': freq_info,
            'target_info': tgt_info,
            'keep_label': True
        }

# ================================================================
# 9. Medium Cardinality (51-150): Label + Frequency
# ================================================================
if medium_card_cols:
    print(f"\n6. Medium Cardinality 변수 처리 (Label + Frequency)")
    
    for col in medium_card_cols:
        print(f"   처리 중: {col}")
        
        # Frequency Encoding
        train_freq, val_freq, test_freq, freq_info = frequency_encode(
            X_train, X_val, X_test_copy, col
        )
        
        freq_col = f'{col}_freq'
        X_trainval.loc[train_idx, freq_col] = train_freq
        X_trainval.loc[val_idx, freq_col] = val_freq
        X_test[freq_col] = test_freq
        new_numerical_cols.append(freq_col)
        
        all_encoding_info[col] = {
            'type': 'medium_card',
            'freq_info': freq_info,
            'keep_label': True
        }

# ================================================================
# 10. Low & Very Low Cardinality: Label Only
# ================================================================
print(f"\n7. Low Cardinality 변수 처리 (Label Only)")
all_low_card = very_low_card_cols + low_card_cols

for col in all_low_card:
    all_encoding_info[col] = {
        'type': 'low_card',
        'keep_label': True
    }
    print(f"   {col}: Label Encoding만 사용")

# ================================================================
# 11. Label Encoding (모든 남은 범주형 변수)
# ================================================================
print(f"\n8. Label Encoding (범주형 변수)")

label_encoders = {}
cat_cardinalities = []

# Train/Val과 Test 데이터를 하나로 합침
X_combined = pd.concat([X_trainval, X_test], axis=0, ignore_index=True)

for col in categorical_cols:
    le = LabelEncoder()
    
    # Train set으로만 fit
    X_train_col = X_trainval.iloc[train_idx][col].astype(str)
    le.fit(X_train_col)
    
    # 전체 데이터 transform
    def safe_transform(x):
        try:
            return le.transform([str(x)])[0]
        except ValueError:
            return len(le.classes_)  # UNKNOWN
    
    X_combined[col] = X_combined[col].astype(str).apply(safe_transform)
    
    label_encoders[col] = le
    cardinality = len(le.classes_) + 1  # +1 for UNKNOWN
    cat_cardinalities.append(cardinality)
    
    print(f"   {col}: {cardinality}개 카테고리")

# 다시 Train/Val과 Test로 분리
X_trainval_encoded = X_combined.iloc[:len(X_trainval)].reset_index(drop=True)
X_test_encoded = X_combined.iloc[len(X_trainval):].reset_index(drop=True)

# ================================================================
# 12. 저장
# ================================================================
print("\n9. 저장 중...")

save_dir = ''

# 12-1. 인코딩된 데이터 저장 (CSV)
X_trainval_encoded[target_col] = y_trainval
X_trainval_encoded.to_csv(f'encoded_trainval_data.csv', index=False)
print(f"   ✓ Train/Val 데이터: encoded_trainval_data.csv")

X_test_encoded[target_col] = y_test
X_test_encoded.to_csv(f'encoded_test_data.csv', index=False)
print(f"   ✓ Test 데이터: encoded_test_data.csv")

# 12-2. 인덱스 저장
np.savez(f'data_split.npz',
         train_idx=train_idx,
         val_idx=val_idx,
         test_size=len(X_test))
print(f"   ✓ 분할 인덱스: data_split.npz")

# 12-3. 메타 정보 저장
metadata = {
    'label_encoders': label_encoders,
    'cat_cardinalities': cat_cardinalities,
    'numerical_cols': new_numerical_cols,
    'categorical_cols': categorical_cols,
    'target_col': target_col,
    'seed': seed,
    'trainval_size': len(X_trainval),
    'test_size': len(X_test),
    'encoding_info': all_encoding_info,
    'original_categorical_cols': categorical_cols + very_high_card_cols,
    'encoding_strategy': {
        'very_high_card': very_high_card_cols,
        'high_card': high_card_cols,
        'medium_card': medium_card_cols,
        'low_card': all_low_card
    }
}

with open(f'preprocessing_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✓ 메타데이터: preprocessing_metadata.pkl")

# ================================================================
# 13. 검증
# ================================================================
print("\n10. 인코딩 검증")

# Target/Frequency encoded 컬럼들의 통계
print("\n   추가된 Feature 통계:")
new_features = [col for col in new_numerical_cols if col not in numerical_cols]
for feat in new_features:
    train_mean = X_trainval_encoded.iloc[train_idx][feat].mean()
    train_std = X_trainval_encoded.iloc[train_idx][feat].std()
    val_mean = X_trainval_encoded.iloc[val_idx][feat].mean()
    test_mean = X_test_encoded[feat].mean()
    
    print(f"   {feat:30s}: Train={train_mean:.4f}±{train_std:.4f}, Val={val_mean:.4f}, Test={test_mean:.4f}")

# ================================================================
# 14. 완료
# ================================================================
print("\n" + "=" * 70)
print("✅ 전처리 완료! (최적 인코딩 전략)")
print("=" * 70)
print(f"\n저장된 파일:")
print(f"  1. encoded_trainval_data.csv")
print(f"  2. encoded_test_data.csv")
print(f"  3. data_split.npz")
print(f"  4. preprocessing_metadata.pkl")
print("\n데이터 구성:")
print(f"  - Train: {len(train_idx):,}개 ({len(train_idx)/len(X_trainval)*100:.1f}%)")
print(f"  - Val:   {len(val_idx):,}개 ({len(val_idx)/len(X_trainval)*100:.1f}%)")
print(f"  - Test:  {len(X_test):,}개 (별도 파일)")
print(f"\n변수 구성:")
print(f"  - 연속형 (원본): {len(numerical_cols)}개")
print(f"  - 연속형 (인코딩 추가): {len(new_numerical_cols)}개")
print(f"  - 범주형 (Label): {len(categorical_cols)}개")
print(f"\n인코딩 전략:")
print(f"  - K-Fold Target (Primary): {len(very_high_card_cols)}개 변수")
print(f"  - Label + Freq + Target:   {len(high_card_cols)}개 변수")
print(f"  - Label + Frequency:       {len(medium_card_cols)}개 변수")
print(f"  - Label Only:              {len(all_low_card)}개 변수")
print("\n✅ Data Leakage 방지:")
print(f"  - K-Fold (n=5) Target Encoding 적용")
print(f"  - Smoothing 정규화 적용 (small sample 대응)")
print(f"  - Out-of-Fold 방식으로 완벽한 분리")
print("=" * 70)
