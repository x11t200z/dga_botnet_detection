import os
import pandas as pd
import glob

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn đến thư mục gốc chứa các thư mục con (alureon, banjori...)
# Lưu ý: Thay đổi đường dẫn này cho đúng với máy của bạn
UMUDGA_ROOT_PATH = r"dataset\UMUDGA - University of Murcia Domain Generation Algorithm Dataset\Fully Qualified Domain Names" 
TRANCO_PATH = os.path.join('dataset', 'tranco_2NP39-1m', 'top-1m.csv')

def load_dga_data(root_path):
    all_dga_domains = []
    
    print(f"Dang quet du lieu tu: {root_path}")
    # Lấy danh sách tất cả thư mục con (tên họ malware)
    families = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    
    print(f"Tim thay {len(families)} ho malware.")
    
    for family in families:
        family_dir = os.path.join(root_path, family)
        list_dir = os.path.join(family_dir, "list")
        
        # Kiểm tra xem thư mục 'list' có tồn tại không
        if not os.path.isdir(list_dir):
            print(f" [!] Khong tim thay thu muc 'list' trong {family}")
            continue
        
        # Tìm tất cả file.txt trong thư mục 'list'
        txt_files = glob.glob(os.path.join(list_dir, "*.txt"))
        
        if not txt_files:
            print(f" [!] Khong tim thay file .txt trong {family}/list")
            continue
        
        target_file = None
        
        # Ưu tiên tìm file có chữ '1000' hoặc '5000' trong tên
        for f in txt_files:
            if '1000' in f:
                target_file = f
                break
        
        # Nếu không thấy file 1000, lấy file txt đầu tiên
        if target_file is None:
            target_file = txt_files[0]
        
        try:
            # Đọc file, mỗi dòng là 1 domain
            df_temp = pd.read_csv(target_file, header=None, names=['domain'])
            
            df_temp['label'] = 1
            df_temp['family'] = family
            
            # Giới hạn tối đa 1000 mẫu mỗi họ
            if len(df_temp) > 1000:
                df_temp = df_temp.sample(1000, random_state=42)
            
            all_dga_domains.append(df_temp)
            
        except Exception as e:
            print(f" [!] Loi doc folder {family}: {e}")
    
    if not all_dga_domains:
        print("Khong tim thay du lieu DGA nao!")
        return pd.DataFrame()
        
    # Gộp tất cả lại
    master_dga = pd.concat(all_dga_domains, ignore_index=True)
    return master_dga

# --- THỰC THI ---
# 1. Tải DGA
df_dga = load_dga_data(UMUDGA_ROOT_PATH)
print(f"\nTong cong DGA domains: {len(df_dga)}")

# 2. Tải Benign (Tranco)
print("\nDang tai Tranco list...")
try:
    df_benign = pd.read_csv(TRANCO_PATH, header=None, names=['rank', 'domain'])
    df_benign = df_benign[['domain']]
    df_benign['label'] = 0
    df_benign['family'] = 'benign'
    
    # Lấy số lượng Benign bằng với số lượng DGA để cân bằng (50-50)
    # Ví dụ: Nếu có 50k DGA, ta lấy 50k Benign top đầu
    limit = len(df_dga)
    df_benign = df_benign.iloc[:limit]
    print(f"Da tai {len(df_benign)} ten mien sach.")
    
except Exception as e:
    print(f"Loi doc file Tranco: {e}")
    df_benign = pd.DataFrame()

# 3. Gộp thành Master Dataset
if not df_dga.empty and not df_benign.empty:
    df_final = pd.concat([df_dga, df_benign], ignore_index=True)
    
    # Xáo trộn
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Chuyển về string và lowercase
    df_final['domain'] = df_final['domain'].astype(str).str.lower()
    
    print("\n--- KET QUA CUOI CUNG ---")
    print(df_final['label'].value_counts())
    
    # Lưu file
    df_final.to_csv('dataset_full.csv', index=False)
    print("\nDa luu 'dataset_full.csv'. Ban da san sang cho buoc trich xuat dac trung!")
else:
    print("\nCo loi xay ra, vui long kiem tra lai duong dan file.")