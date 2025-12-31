import pandas as pd
# Import file utils vừa tạo
import feature_utils 

print("Dang doc du lieu 'dataset_full.csv'...")
try:
    df = pd.read_csv('dataset_full.csv')
    df['domain'] = df['domain'].astype(str)
except FileNotFoundError:
    print("Khong tim thay file dataset!")
    exit()

# Hàm wrapper để tương thích với pandas apply
def extract_wrapper(row):
    domain = str(row['domain'])
    # Gọi hàm từ feature_utils
    feat_dict = feature_utils.get_features_dict(domain)
    # Trả về list theo đúng thứ tự cột đã định nghĩa
    return [feat_dict[col] for col in feature_utils.FEATURE_NAMES]

# --- TRÍCH XUẤT VÀ LƯU KẾT QUẢ ---
print("Dang tinh toan dac trung (Su dung module feature_utils)...")

# Sử dụng tên cột từ utils để đảm bảo đồng nhất
cols = feature_utils.FEATURE_NAMES
print(f"Tổng số features: {len(cols)}")

features_list = df.apply(extract_wrapper, axis=1).tolist()
df_features = pd.DataFrame(features_list, columns=cols)
df_final = pd.concat([df, df_features], axis=1)

# Lưu file (6 features + label)
cols_to_keep = cols + ['label']
df_final[cols_to_keep].to_csv('dataset_training_ready.csv', index=False)

print(f"\nDa luu 'dataset_training_ready.csv' voi {len(cols)} dac trung va label.")
print(f"Kich thuoc du lieu: {df_final.shape}")
print("\nHay chay file Train ngay!")