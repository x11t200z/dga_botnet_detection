import pandas as pd
import joblib
import feature_utils
import sys

# 1. Load dữ liệu
print("1. Dang tai du lieu...")
try:
    # Load file đặc trưng (để dự đoán)
    df_features = pd.read_csv('dataset_training_ready.csv')
    
    # Load file gốc (để lấy tên miền text và họ malware)
    df_raw = pd.read_csv('dataset_full.csv')
    
    # KIỂM TRA AN TOÀN: Hai file phải khớp số dòng
    if len(df_features) != len(df_raw):
        print(f"[!] LOI: File dataset_full ({len(df_raw)}) va training_ready ({len(df_features)}) khong khop so luong!")
        print("    Vui long chay lai file 2_feature_extration.py de dong bo.")
        sys.exit()

    # --- [CẬP NHẬT] GHÉP THÊM CỘT FAMILY ---
    df_features['domain'] = df_raw['domain']
    df_features['family'] = df_raw['family'] # Lấy thêm thông tin họ malware/benign
    
    print(f" -> Da tai {len(df_features)} mau du lieu.")

except FileNotFoundError as e:
    print(f"[!] Loi: Khong tim thay file du lieu. {e}")
    sys.exit()

# 2. Load Model
print("2. Dang tai Model...")
try:
    model = joblib.load('dga_xgboost_model.pkl')
except FileNotFoundError:
    print("[!] Khong tim thay file model 'dga_xgboost_model.pkl'")
    sys.exit()

# 3. Thực hiện dự đoán
print("3. Dang phan tich loi...")

# --- [QUAN TRỌNG] LOẠI BỎ CÁC CỘT KHÔNG PHẢI FEATURE ---
# Model chỉ hiểu các con số đặc trưng, nếu để lọt 'domain' hay 'family' vào sẽ gây lỗi
cols_to_drop = ['label', 'domain', 'family'] 
X = df_features.drop(columns=cols_to_drop) 
y_true = df_features['label']

# Dự đoán
y_pred = model.predict(X)

# Tạo DataFrame chứa các mẫu bị sai
errors = df_features[y_true != y_pred].copy()
errors['pred'] = y_pred[y_true != y_pred]

# Map label số sang tên cho dễ đọc
label_map = {0: 'Sach (Benign)', 1: 'Doc (DGA)'}
errors['label_name'] = errors['label'].map(label_map)
errors['pred_name'] = errors['pred'].map(label_map)

# 4. Hiển thị kết quả
print(f"\n>>> TONG SO MAU SAI: {len(errors)} / {len(df_features)} (Ty le sai: {len(errors)/len(df_features)*100:.2f}%)")

# Định nghĩa các cột muốn hiển thị ra màn hình
display_cols = ['domain', 'family', 'length', 'entropy', 'meaningful_ratio', 'bigram_score']

print("\n" + "="*80)
print("TRUONG HOP 1: FALSE POSITIVE (BAO DONG GIA)")
print("Thuc te la SACH (Family=benign), nhung Model bao nham la DGA.")
print("-" * 80)
# Lọc ra các mẫu Label=0 (Sạch) nhưng Pred=1 (DGA)
fp_df = errors[errors['label'] == 0]
if not fp_df.empty:
    # Sắp xếp theo entropy giảm dần để xem cái nào "loạn" nhất
    print(fp_df[display_cols].sort_values(by='entropy', ascending=False).head(15).to_string(index=False))
else:
    print("Khong co mau nao!")

print("\n" + "="*80)
print("TRUONG HOP 2: FALSE NEGATIVE (LOT LUOI)")
print("Thuc te la DGA, nhung Model bao la SACH.")
print("Need check: Hay xem cot 'family' de biet con Botnet nao ghe gom nhat!")
print("-" * 80)
# Lọc ra các mẫu Label=1 (DGA) nhưng Pred=0 (Sạch)
fn_df = errors[errors['label'] == 1]
if not fn_df.empty:
    # Sắp xếp theo meaningful_ratio giảm dần (những con giả dạng giống từ điển nhất)
    print(fn_df[display_cols].sort_values(by='meaningful_ratio', ascending=False).head(15).to_string(index=False))
    
    print("-" * 30)
    print("TOP 5 HO MALWARE BI BO SOT NHIEU NHAT:")
    print(fn_df['family'].value_counts().head(5))
else:
    print("Khong co mau nao!")