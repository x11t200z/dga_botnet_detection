import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Thiết lập không gian tham số để tìm kiếm
# Random Forest có nhiều tham số, việc tinh chỉnh chúng giúp tăng 2-5% accuracy
param_dist = {
    # Giảm số lượng cây xuống để chạy nhanh hơn (100-200 là điểm vàng)
    'n_estimators': [50, 100, 200],
    
    # Giới hạn độ sâu để tránh học vẹt và tràn RAM
    'max_depth': [10, 20, 30],
    
    # Tăng nhẹ min_samples để mô hình tổng quát hơn
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 8],
    
    # Luôn dùng True cho nhanh (False rất chậm với dữ liệu lớn)
    'bootstrap': [True] 
}

print("\n--- BAT DAU TIM KIEM THAM SO TOI UU (TUNING) ---")
print("Qua trinh nay se chay thu nghiem nhieu cau hinh khac nhau...")

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Chạy tìm kiếm ngẫu nhiên (nhanh hơn GridSearch)
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,          # Thử ngẫu nhiên 50 tổ hợp
    cv=5,               # Cross-validation 5 lần
    verbose=2,
    random_state=42,
    n_jobs=-1,           # Dùng 100% CPU
    scoring='f1'
)

start_time = time.time()
rf_random.fit(X_train, y_train)
end_time = time.time()

print(f"\n[DONE] Tuning hoan tat trong {end_time - start_time:.2f}s")
print(f"Tham so tot nhat: {rf_random.best_params_}")

# 3. Lấy model tốt nhất để đánh giá
best_model = rf_random.best_estimator_

print("\n--- KET QUA DANH GIA (BEST MODEL) ---")
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Do chinh xac (Accuracy): {accuracy * 100:.2f}%")
print("-" * 30)
print(classification_report(y_test, y_pred, target_names=['Legit', 'DGA']))

# 4. Phân tích đặc trưng quan trọng
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nTop 5 Dac trung quan trong nhat:")
for i in range(5):
    print(f"{i+1}. {X.columns[indices[i]]} ({importances[indices[i]]:.4f})")

# 5. Lưu model
joblib.dump(best_model, 'dga_rf_model.pkl')
print("\nDa luu model toi uu vao 'dga_rf_model.pkl'")

# 5. Đánh giá kết quả chi tiết
print("\n--- PHAN TICH CHI TIET (EVALUATION) ---")
y_pred = best_model.predict(X_test)

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"-> Do chinh xac (Accuracy): {accuracy * 100:.2f}%")

# In báo cáo phân loại (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legit (Sach)', 'DGA (Doc)']))

# --- PHẦN THÊM MỚI: CONFUSION MATRIX ---
print("-" * 30)
print("MA TRAN NHAM LAN (CONFUSION MATRIX):")
cm = confusion_matrix(y_test, y_pred)

# Lấy các giá trị cụ thể
tn, fp, fn, tp = cm.ravel()

print(f" [+] True Negative (Sach du doan dung): {tn} mau")
print(f" [!] False Positive (Sach bi nham la DGA): {fp} mau  <-- (Luu y cai nay)")
print(f" [!] False Negative (DGA bi lot luoi):     {fn} mau")
print(f" [+] True Positive (DGA du doan dung):    {tp} mau")
print("-" * 30)

# 6. Vẽ biểu đồ Heatmap (để lưu vào báo cáo)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d',           # Hiển thị số nguyên
    cmap='Blues',      # Tông màu xanh
    xticklabels=['Du doan: Sach', 'Du doan: DGA'],
    yticklabels=['Thuc te: Sach', 'Thuc te: DGA']
)
plt.title(f'Confusion Matrix - Accuracy: {accuracy*100:.2f}%')
plt.ylabel('Nhan thuc te (Actual)')
plt.xlabel('Nhan du doan (Predicted)')
plt.tight_layout()
plt.savefig('confusion_matrix_result.png') # Lưu ảnh
plt.show()