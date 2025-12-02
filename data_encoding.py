import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import sklearn.model_selection

# ================================================================
# 1. 데이터 로드
# ================================================================
print("=" * 70)
print("데이터 전처리 및 저장")
print("=" * 70)

# 1번 CSV: Train/Validation용
df_trainval = pd.read_csv('cd_trainset.csv')
print(f"\n1. 데이터 로드")
print(f"   Train/Val 파일: {df_trainval.shape}")

# 2번 CSV: Test용
df_test = pd.read_csv('cd_testset.csv')  # 실제 테스트 파일명으로 수정 필요
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
train_size = int(len(X_trainval) * 0.9)
val_size = len(X_trainval) - train_size

train_idx, val_idx = sklearn.model_selection.train_test_split(
    np.arange(len(X_trainval)), test_size=0.1, random_state=seed
)

print(f"   Train: {len(train_idx)} samples ({len(train_idx)/len(X_trainval)*100:.1f}%)")
print(f"   Val:   {len(val_idx)} samples ({len(val_idx)/len(X_trainval)*100:.1f}%)")
print(f"   Test:  {len(X_test)} samples (별도 파일)")

# ================================================================
# 3. 범주형 변수 인코딩 (Train 기준)
# ================================================================
print("\n3. 범주형 변수 인코딩 (Train 기준)")

label_encoders = {}
cat_cardinalities = []

# Train/Val 데이터와 Test 데이터를 하나로 합침 (인코딩을 위해)
X_combined = pd.concat([X_trainval, X_test], axis=0, ignore_index=True)

for col in categorical_cols:
    le = LabelEncoder()
    
    # Train set으로만 fit
    X_train_col = X_trainval.iloc[train_idx][col].astype(str)
    le.fit(X_train_col)
    
    # 전체 데이터 transform (Train/Val + Test)
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
# 4. 저장
# ================================================================
print("\n4. 저장 중...")

save_dir = '/mnt/user-data/outputs/'

# 4-1. 인코딩된 데이터 저장 (CSV)
# Train/Val 데이터
X_trainval_encoded['PROC_EXPOSE_LOG'] = y_trainval
X_trainval_encoded.to_csv(f'encoded_trainval_data.csv', index=False)
print(f"   ✓ Train/Val 데이터: {save_dir}encoded_trainval_data.csv")

# Test 데이터
X_test_encoded['PROC_EXPOSE_LOG'] = y_test
X_test_encoded.to_csv(f'encoded_test_data.csv', index=False)
print(f"   ✓ Test 데이터: {save_dir}encoded_test_data.csv")

# 4-2. 인덱스 저장
np.savez(f'data_split.npz',
         train_idx=train_idx,
         val_idx=val_idx,
         test_size=len(X_test))
print(f"   ✓ 분할 인덱스: {save_dir}data_split.npz")

# 4-3. 메타 정보 저장
metadata = {
    'label_encoders': label_encoders,
    'cat_cardinalities': cat_cardinalities,
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'target_col': target_col,
    'seed': seed,
    'trainval_size': len(X_trainval),
    'test_size': len(X_test)
}

with open(f'preprocessing_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✓ 메타데이터:preprocessing_metadata.pkl")

# ================================================================
# 5. 완료
# ================================================================
print("\n" + "=" * 70)
print("✅ 전처리 완료!")
print("=" * 70)
print(f"\n저장된 파일:")
print(f"  1. {save_dir}encoded_trainval_data.csv  (Train/Val 인코딩 데이터)")
print(f"  2. {save_dir}encoded_test_data.csv      (Test 인코딩 데이터)")
print(f"  3. {save_dir}data_split.npz              (분할 인덱스)")
print(f"  4. {save_dir}preprocessing_metadata.pkl  (메타정보)")
print("\n데이터 구성:")
print(f"  - Train: {len(train_idx):,}개 ({len(train_idx)/len(X_trainval)*100:.1f}%)")
print(f"  - Val:   {len(val_idx):,}개 ({len(val_idx)/len(X_trainval)*100:.1f}%)")
print(f"  - Test:  {len(X_test):,}개 (별도 파일)")
print("=" * 70)


# (tabm) suyeol@node4:~/project/tabm$ python data_encoding.py 
# ======================================================================
# 데이터 전처리 및 저장
# ======================================================================

# 1. 데이터 로드
#    Train/Val 파일: (416247, 21)
#    Test 파일: (52905, 21)
#    연속형 변수: 6개
#    범주형 변수: 14개

# 2. 데이터 분할 (Train 90% / Val 10%)
#    Train: 374622 samples (90.0%)
#    Val:   41625 samples (10.0%)
#    Test:  52905 samples (별도 파일)

# 3. 범주형 변수 인코딩 (Train 기준)
#    PR_NAME: 10개 카테고리
#    WAVE_LENGTH: 3개 카테고리
#    R_VENDOR: 7개 카테고리
#    TRACK_RECIPE: 103개 카테고리
#    ROUTE_DESC: 140개 카테고리
#    PROC_EQ: 40개 카테고리
#    DEV_ID: 2897개 카테고리
#    PROCESS_ID: 76개 카테고리
#    EXPOSE_TYPE: 4개 카테고리
#    EQ_TYPE: 4개 카테고리
#    ROUTE_PREFIX: 352개 카테고리
#    RETICLE_SUBFIX: 110개 카테고리
#    RETICLE_PREFIX: 163개 카테고리
#    LENGTH_CTYPE: 5개 카테고리

# 4. 저장 중...
#    ✓ Train/Val 데이터: /mnt/user-data/outputs/encoded_trainval_data.csv
#    ✓ Test 데이터: /mnt/user-data/outputs/encoded_test_data.csv
#    ✓ 분할 인덱스: /mnt/user-data/outputs/data_split.npz
#    ✓ 메타데이터:preprocessing_metadata.pkl

# ======================================================================
# ✅ 전처리 완료!
# ======================================================================

# 저장된 파일:
#   1. /mnt/user-data/outputs/encoded_trainval_data.csv  (Train/Val 인코딩 데이터)
#   2. /mnt/user-data/outputs/encoded_test_data.csv      (Test 인코딩 데이터)
#   3. /mnt/user-data/outputs/data_split.npz              (분할 인덱스)
#   4. /mnt/user-data/outputs/preprocessing_metadata.pkl  (메타정보)

# 데이터 구성:
#   - Train: 374,622개 (90.0%)
#   - Val:   41,625개 (10.0%)
#   - Test:  52,905개 (별도 파일)
# ======================================================================
