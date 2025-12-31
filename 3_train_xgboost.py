import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

# 1. Đọc dữ liệu
print("Dang tai du lieu...")
try:
    df = pd.read_csv('dataset_training_ready.csv')
except FileNotFoundError:
    print("Loi: Khong tim thay file dataset_training_ready.csv")
    exit()

X = df.drop(columns=['label'])
y = df['label']

print(f"Features ({len(X.columns)}): {list(X.columns)}")

# Chia tập dữ liệu (80train, 20test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Cấu hình XGBoost
print("\n--- BAT DAU HUAN LUYEN XGBOOST ---")

# Thiết lập bộ tham số để tìm kiếm (Tuning)
param_dist = {
    'n_estimators': [100, 200, 300],        # Số lượng cây (nhiều hơn thường tốt hơn)
    'learning_rate': [0.01, 0.1, 0.2],      # Tốc độ học (thấp = kỹ, cao = nhanh)
    'max_depth': [4, 6, 8, 10],             # Độ sâu của cây (sâu quá dễ overfit)
    'colsample_bytree': [0.7, 1.0],         # Lấy mẫu đặc trưng (tránh học vẹt)
    'subsample': [0.7, 0.8, 1.0]            # Lấy mẫu dữ liệu
}

xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    # Đã xóa dòng use_label_encoder
    random_state=42,
    n_jobs=-1
)

# Dùng RandomizedSearchCV để tìm cấu hình tốt nhất
random_search = RandomizedSearchCV(
    xgb_clf, 
    param_distributions=param_dist, 
    n_iter=20,          # Thử 20 tổ hợp ngẫu nhiên (chạy khá nhanh)
    scoring='accuracy', 
    n_jobs=-1, 
    cv=3, 
    verbose=1,
    random_state=42
)

start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()

print(f"\n[DONE] Training hoan tat trong {end_time - start_time:.2f}s")
best_model = random_search.best_estimator_
print(f"Tham so tot nhat: {random_search.best_params_}")

# 3. Đánh giá kết quả
print("\n--- KET QUA DANH GIA (XGBOOST) ---")
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Do chinh xac (Accuracy): {accuracy * 100:.2f}%")
print("-" * 30)
print(classification_report(y_test, y_pred, target_names=['Legit', 'DGA']))

# 4. Phân tích đặc trưng quan trọng (Feature Importance)
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nTop Dac trung quan trong nhat (XGBoost):")
for i in range(len(X.columns)):
    print(f"{i+1}. {X.columns[indices[i]]} ({importances[indices[i]]:.4f})")

# 5. Vẽ Confusion Matrix
print("\n--- MA TRAN NHAM LAN ---")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"True Negative (Sach dung): {tn} | False Positive (Bao nham): {fp}")
print(f"False Negative (Sot DGA):  {fn} | True Positive (Bat dung):  {tp}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Sach', 'DGA'], yticklabels=['Sach', 'DGA'])
plt.title(f'XGBoost Confusion Matrix - Acc: {accuracy*100:.2f}%')
plt.ylabel('Thuc te')
plt.xlabel('Du doan')
plt.savefig('confusion_matrix_xgboost.png')
plt.close()

# 6. Lưu model (Vẫn lưu tên cũ để file demo không phải sửa code)
joblib.dump(best_model, 'dga_rf_model.pkl') 
print("\nDa luu model (XGBoost) vao file 'dga_rf_model.pkl'")