import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder
import time

# ================================================================
# 데이터 전처리 및 저장 (최적화 + Val=Test 버전)
# Train: cd_trainset 전체, Val=Test: cd_testset 동일
# ================================================================
print("=" * 70)
print("데이터 전처리 및 저장 (Val=Test 버전)")
print("=" * 70)

start_total = time.time()

# ================================================================
# 1. 데이터 로드
# ================================================================
print("\n1. 데이터 로드")

df_trainset = pd.read_csv('cd_trainset.csv')
print(f"   Trainset 파일: {df_trainset.shape}")

df_testset = pd.read_csv('cd_testset.csv')
print(f"   Testset 파일: {df_testset.shape}")

# 타겟 분리
target_col = 'PROC_EXPOSE_LOG'

# Feature 타입 분리
numerical_cols = df_trainset.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove(target_col)  # 타겟 제거
categorical_cols = df_trainset.select_dtypes(include=['object']).columns.tolist()

print(f"   연속형 변수: {len(numerical_cols)}개")
print(f"   범주형 변수: {len(categorical_cols)}개")

# ================================================================
# 2. 데이터 분할 전략
# ================================================================
print("\n2. 데이터 분할 전략")
print("   Train: cd_trainset 전체")
print("   Val:   cd_testset 전체 (Test와 동일)")
print("   Test:  cd_testset 전체 (Val과 동일)")

# Train 인덱스 (trainset 전체)
train_idx = np.arange(len(df_trainset))

# Val/Test 인덱스 (testset 전체, 동일)
# trainset에는 없지만 나중에 data_numpy에서 testset을 val과 test 둘 다로 사용
val_idx = np.array([])  # 더미 (실제로는 testset 사용)

print(f"\n   분할 결과:")
print(f"   - Train: {len(train_idx):,}개 (cd_trainset 전체)")
if 'DATE' in df_trainset.columns:
    print(f"     기간: {df_trainset['DATE'].min()} ~ {df_trainset['DATE'].max()}")
print(f"   - Val:   {len(df_testset):,}개 (cd_testset 전체)")
print(f"   - Test:  {len(df_testset):,}개 (cd_testset 전체, Val과 동일)")
if 'DATE' in df_testset.columns:
    print(f"     기간: {df_testset['DATE'].min()} ~ {df_testset['DATE'].max()}")

# ================================================================
# 3. 범주형 변수 인코딩 (최적화 버전 - 벡터화)
# ================================================================
print("\n3. 범주형 변수 인코딩 (벡터화 방식)")
start_cat = time.time()

# Train/Test 데이터 준비
X_trainset = df_trainset.drop(columns=[target_col])
y_trainset = df_trainset[target_col].values

X_testset = df_testset.drop(columns=[target_col])
y_testset = df_testset[target_col].values

# 범주형 변수만 추출
X_trainset_cat = X_trainset[categorical_cols].values.astype(str)
X_testset_cat = X_testset[categorical_cols].values.astype(str)

# Train set으로만 fit (trainset 전체 사용)
X_train_cat = X_trainset_cat[train_idx]

# OrdinalEncoder 사용 (벡터화!)
unknown_value = np.iinfo('int64').max - 3
encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=unknown_value,
    dtype='int64'
)
encoder.fit(X_train_cat)

# 전체 데이터 transform
X_trainset_cat_encoded = encoder.transform(X_trainset_cat)
X_testset_cat_encoded = encoder.transform(X_testset_cat)

# UNKNOWN 재매핑
max_values = X_trainset_cat_encoded[train_idx].max(axis=0)

# Trainset에서 UNKNOWN 처리
for col_idx in range(X_trainset_cat_encoded.shape[1]):
    mask = X_trainset_cat_encoded[:, col_idx] == unknown_value
    X_trainset_cat_encoded[mask, col_idx] = max_values[col_idx] + 1

# Testset에서 UNKNOWN 처리
for col_idx in range(X_testset_cat_encoded.shape[1]):
    mask = X_testset_cat_encoded[:, col_idx] == unknown_value
    X_testset_cat_encoded[mask, col_idx] = max_values[col_idx] + 1

# DataFrame에 다시 넣기
for i, col in enumerate(categorical_cols):
    X_trainset[col] = X_trainset_cat_encoded[:, i]
    X_testset[col] = X_testset_cat_encoded[:, i]

# Cardinality 계산
cat_cardinalities = []
for col_idx in range(len(categorical_cols)):
    n_categories = int(max_values[col_idx] + 2)  # +2 for UNKNOWN
    cat_cardinalities.append(n_categories)
    print(f"   {categorical_cols[col_idx]}: {n_categories}개 카테고리")

elapsed_cat = time.time() - start_cat
print(f"\n   ✓ 범주형 인코딩 완료 ({elapsed_cat:.2f}초)")

# ================================================================
# 4. 저장
# ================================================================
print("\n4. 저장 중...")
save_dir = ''

# 4-1. 인코딩된 데이터 저장 (CSV)
# Trainset → encoded_trainval_data.csv (이름은 유지하되 trainset 전체)
X_trainset['PROC_EXPOSE_LOG'] = y_trainset
X_trainset.to_csv(f'{save_dir}encoded_train_data.csv', index=False)
print(f"   ✓ Train 데이터: {save_dir}encoded_train_data.csv")

# Testset → encoded_test_data.csv (Val과 Test에서 공유)
X_testset['PROC_EXPOSE_LOG'] = y_testset
X_testset.to_csv(f'{save_dir}encoded_valtest_data.csv', index=False)
print(f"   ✓ Val/Test 데이터: {save_dir}encoded_valtest_data.csv")

# 4-2. 인덱스 저장
# train_idx: trainset 전체 인덱스
# val_idx: 빈 배열 (실제로는 testset 전체 사용)
# test_size: testset 크기
np.savez(f'{save_dir}data_split.npz',
         train_idx=train_idx,
         val_idx=val_idx,
         test_size=len(X_testset))
print(f"   ✓ 분할 인덱스: {save_dir}data_split.npz")

# 4-3. 메타 정보 저장
metadata = {
    'encoder': encoder,
    'cat_cardinalities': cat_cardinalities,
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'target_col': target_col,
    'split_type': 'val_equals_test',  # Val=Test 분할 표시
    'trainset_size': len(X_trainset),
    'testset_size': len(X_testset)
}

with open(f'{save_dir}preprocessing_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✓ 메타데이터: {save_dir}preprocessing_metadata.pkl")

# ================================================================
# 5. 완료
# ================================================================
elapsed_total = time.time() - start_total

print("\n" + "=" * 70)
print("✅ 전처리 완료!")
print("=" * 70)
print(f"\n저장된 파일:")
print(f"  1. {save_dir}encoded_trainval_data.csv (Train 데이터)")
print(f"  2. {save_dir}encoded_test_data.csv (Val/Test 데이터, 동일)")
print(f"  3. {save_dir}data_split.npz")
print(f"  4. {save_dir}preprocessing_metadata.pkl")
print(f"\n데이터 구성:")
print(f"  - Train: {len(train_idx):,}개 (cd_trainset 전체)")
print(f"  - Val:   {len(X_testset):,}개 (cd_testset 전체)")
print(f"  - Test:  {len(X_testset):,}개 (cd_testset 전체, Val과 동일)")
print(f"\n⏱️  처리 시간:")
print(f"  - 범주형 인코딩: {elapsed_cat:.2f}초")
print(f"  - 전체 시간: {elapsed_total:.2f}초")
print(f"\n✅ Val과 Test가 동일한 unseen 데이터셋 사용!")
print("=" * 70)
