import pandas as pd
import numpy as np
import feature_utils
import multiprocessing
from tqdm import tqdm
import os

# --- CẤU HÌNH ---
N_JOBS = multiprocessing.cpu_count()  # Sử dụng tất cả các cores
FEATURE_COLUMNS = feature_utils.FEATURE_NAMES

def process_chunk(chunk):
    """Xử lý một phần của DataFrame."""
    # Quan trọng: Mỗi process con không cần reload feature_utils, 
    # nhưng global variables trong feature_utils sẽ được copy.
    # Đảm bảo dictionary được load
    if not feature_utils.COMMON_WORDS_SET:
        feature_utils.load_google_10k()
        
    results = []
    for domain in chunk:
        domain = str(domain)
        feat_dict = feature_utils.get_features_dict(domain)
        results.append([feat_dict[col] for col in FEATURE_COLUMNS])
    return results

if __name__ == '__main__':
    print(f"Dang doc du lieu 'dataset_full.csv'...")
    try:
        df = pd.read_csv('dataset_full.csv')
        df['domain'] = df['domain'].astype(str)
    except FileNotFoundError:
        print("Khong tim thay file dataset! Hay chay 1_load.py truoc.")
        exit()

    print(f"Bat dau trich xuat dac trung voi {N_JOBS} luong (cores)...")
    
    # Chia dữ liệu thành các chunks
    domains = df['domain'].tolist()
    chunk_size = len(domains) // N_JOBS + 1
    chunks = [domains[i:i + chunk_size] for i in range(0, len(domains), chunk_size)]
    
    # Xử lý song song
    with multiprocessing.Pool(processes=N_JOBS) as pool:
        # Sử dụng tqdm để hiện progress bar
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), unit="chunk"))
    
    # Gộp kết quả
    flat_results = [item for sublist in results for item in sublist]
    
    print("\nDang luu ket qua...")
    df_features = pd.DataFrame(flat_results, columns=FEATURE_COLUMNS)
    df_final = pd.concat([df, df_features], axis=1)

    cols_to_keep = FEATURE_COLUMNS + ['label']
    
    # Convert label to int if needed (usually it is already)
    # Save
    output_file = 'dataset_training_ready.csv'
    df_final[cols_to_keep].to_csv(output_file, index=False)

    print(f"\n[OK] Da luu '{output_file}'")
    print(f"Tong so features: {len(FEATURE_COLUMNS)}")
    print(f"Kich thuoc du lieu: {df_final.shape}")