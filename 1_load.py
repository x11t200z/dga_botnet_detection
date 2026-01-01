import os
import pandas as pd
import glob

# --- CẤU HÌNH ---
UMUDGA_ROOT_PATH = r"dataset\UMUDGA - University of Murcia Domain Generation Algorithm Dataset\Fully Qualified Domain Names" 
TRANCO_PATH = os.path.join('dataset', 'tranco_2NP39-1m', 'top-1m.csv')

# Số lượng mẫu cho mỗi họ Malware, tổng có 50 họ
SAMPLES_PER_FAMILY = 10000

def load_dga_data(root_path):
    all_dga_domains = []
    print(f"Dang quet du lieu tu: {root_path}")
    families = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    print(f"Tim thay {len(families)} ho malware.")
    
    for family in families:
        family_dir = os.path.join(root_path, family)
        list_dir = os.path.join(family_dir, "list")
        
        if not os.path.isdir(list_dir): continue
        
        txt_files = glob.glob(os.path.join(list_dir, "*.txt"))
        if not txt_files: continue
        
        # Ưu tiên tìm file lớn nhất (thường chứa nhiều domain nhất)
        target_file = max(txt_files, key=os.path.getsize)
        
        try:
            df_temp = pd.read_csv(target_file, header=None, names=['domain'])
            df_temp['label'] = 1
            df_temp['family'] = family
            
            if len(df_temp) > SAMPLES_PER_FAMILY:
                df_temp = df_temp.sample(SAMPLES_PER_FAMILY, random_state=42)
            
            all_dga_domains.append(df_temp)
            print(f" - {family}: lay {len(df_temp)} mau.")
            
        except Exception as e:
            print(f" [!] Loi doc folder {family}: {e}")
    
    if not all_dga_domains:
        return pd.DataFrame()
        
    master_dga = pd.concat(all_dga_domains, ignore_index=True)
    return master_dga
def is_valid_benign(domain):
    d = str(domain).lower()
    # Loại bỏ domain quá ngắn (dễ gây nhiễu)
    if len(d) < 5: return False
    # Loại bỏ Punycode gây nhiễu
    if 'xn--' in d: return False
    # Loại bỏ các đuôi CDN/Cloud phổ biến làm model học sai
    if 'cloudfront' in d or 'amazonaws' in d or 'akamai' in d or 'azure' in d:
        return False
    # Loại bỏ domain là IP
    if d.replace('.', '').isdigit(): return False
    return True
# --- THỰC THI ---
# Tải DGA
df_dga = load_dga_data(UMUDGA_ROOT_PATH)
print(f"\nTong cong DGA domains: {len(df_dga)}")

# 2. Tải benign
print("\nDang tai Tranco list...")
try:
    # Đọc file CSV
    df_benign = pd.read_csv(TRANCO_PATH, header=None, names=['rank', 'domain'])
    df_benign = df_benign[df_benign['domain'].apply(is_valid_benign)]
    # [QUAN TRỌNG - THAY ĐỔI Ở ĐÂY]
    # Thay vì lấy ngẫu nhiên (sample), ta chỉ lấy top đầu.
    # Top 50,000 domain đầu tiên chắc chắn là domain xịn, không phải rác.
    
    # Giới hạn số lượng bằng với số lượng DGA để cân bằng
    limit = len(df_dga)
    
    # Lấy đúng limit dòng đầu tiên
    df_benign = df_benign.iloc[:limit] 

    df_benign = df_benign[['domain']]
    df_benign['label'] = 0
    df_benign['family'] = 'benign'
    
    print(f"Da lay {len(df_benign)} ten mien sach")
    
except Exception as e:
    print(f"Loi doc file Tranco: {e}")
    df_benign = pd.DataFrame()

# 3. Gộp và Lưu
if not df_dga.empty and not df_benign.empty:
    df_final = pd.concat([df_dga, df_benign], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final['domain'] = df_final['domain'].astype(str).str.lower()
    
    print("\n--- KET QUA CUOI CUNG ---")
    print(df_final['label'].value_counts())
    
    df_final.to_csv('dataset_full.csv', index=False)
    print("\n[OK] Da luu 'dataset_full.csv'")
else:
    print("\nCo loi xay ra.")