import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib

# 1. Đọc dữ liệu đặc trưng
print("Dang tai du lieu huan luyen...")
try:
    df = pd.read_csv('dataset_training_ready.csv')
except FileNotFoundError:
    print("Loi: Khong tim thay file 'dataset_training_ready.csv'.")
    exit()

# Tách Feature (X) và Label (y)
X = df.drop(columns=['label'])
y = df['label']

print(f"Tong so mau: {len(df)}")
print(f"So luong dac trung: {len(X.columns)} {list(X.columns)}")

# 2. Chia tập dữ liệu (80% Train - 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Cấu hình mô hình XGBoost chạy trên GPU (RTX 4050)
print("\nDang khoi tao mo hinh XGBoost voi GPU...")
model = xgb.XGBClassifier(
    n_estimators=500,        # Số lượng cây quyết định (càng nhiều càng kỹ, nhưng lâu)
    learning_rate=0.05,      # Tốc độ học
    max_depth=12,            # Độ sâu của cây (sâu quá dễ bị học vẹt)
    subsample=0.8,           # Mỗi cây chỉ học trên 80% dữ liệu (tránh học vẹt)
    colsample_bytree=0.8,    # Mỗi cây chỉ dùng 80% đặc trưng
    tree_method='hist',      # Phương pháp tối ưu cho dữ liệu lớn
    device='cuda',           # QUAN TRỌNG: Dòng này kích hoạt GPU RTX 4050
    eval_metric='logloss',
    early_stopping_rounds=20 # Dừng nếu không học tốt hơn sau 20 vòng
)

# 4. Huấn luyện
print("Bat dau huan luyen...")
start_time = time.time()

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

end_time = time.time()
print(f"\n---> Huan luyen xong trong: {end_time - start_time:.2f} giay!")

# 5. Đánh giá kết quả
print("\n--- KET QUA DANH GIA ---")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Do chinh xac (Accuracy): {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Legit Domain', 'DGA Domain']
))

# 6. Biểu đồ Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Du doan: Legit', 'Du doan: DGA'],
    yticklabels=['Thuc te: Legit', 'Thuc te: DGA']
)

plt.title('Confusion Matrix (Ma tran nham lan)')
plt.ylabel('Thuc te')
plt.xlabel('Du doan')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 7. Biểu đồ Feature Importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(
    model,
    importance_type='weight',
    title='Feature Importance (Muc do quan trong dac trung)'
)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# 8. Lưu mô hình
joblib.dump(model, 'dga_detection_model.pkl')
print("\nDa luu mo hinh vao file 'dga_detection_model.pkl'")
