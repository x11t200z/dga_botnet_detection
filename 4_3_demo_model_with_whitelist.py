import pandas as pd
import joblib
import warnings
import os
import sys
import time
import logging

# Tắt cảnh báo Scapy khi khởi động
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

try:
    from scapy.all import rdpcap, DNS, DNSQR, IP
except ImportError:
    print("[!] LOI: Thieu thu vien 'scapy'. Vui long cai dat: pip install scapy")
    sys.exit()

# Import bộ não xử lý đặc trưng
try:
    import feature_utils
except ImportError:
    print("[!] LOI: Khong tim thay file 'feature_utils.py'.")
    sys.exit()

# Tắt các cảnh báo
warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
MODEL_PATH = 'dga_xgboost_model.pkl'
TRANCO_PATH = os.path.join('dataset', 'tranco_2NP39-1m', 'top-1m.csv')

def print_banner():
    print("\n" + "="*70)
    print("   HE THONG PHAT HIEN DGA BOTNET (HYBRID: WHITELIST + XGBOOST)")
    print("   Ho tro: Nhap tay (Manual) & Phan tich File Traffic (.pcap)")
    print("="*70)

def load_system():
    print("[*] Dang khoi dong he thong...")
    
    # 1. Tải Model
    if not os.path.exists(MODEL_PATH):
        print(f"[!] LOI: Khong tim thay model '{MODEL_PATH}'.")
        sys.exit()
    
    print(f" -> Dang tai model...", end=" ")
    try:
        model = joblib.load(MODEL_PATH)
        print("[OK]")
    except Exception as e:
        print(f"\n[!] Loi tai model: {e}")
        sys.exit()

    # 2. Tải Whitelist
    whitelist = set()
    print(f" -> Dang tai Whitelist (Top 1 Trieu websites)...", end=" ")
    if os.path.exists(TRANCO_PATH):
        try:
            df_wl = pd.read_csv(TRANCO_PATH, header=None, names=['rank', 'domain'])
            whitelist = set(df_wl['domain'].astype(str).str.lower())
            print(f"[OK] - Da nap {len(whitelist)} ten mien uy tin.")
        except Exception as e:
            print(f"\n[!] Khong doc duoc whitelist: {e}")
    else:
        print("\n[!] Khong tim thay file Tranco. He thong se chi dung model.")
    
    return model, whitelist

# --- HÀM XỬ LÝ CỐT LÕI (DÙNG CHUNG) ---
def analyze_single_domain(raw_domain, model, whitelist, src_ip="Manual Input"):
    """
    Hàm này nhận vào 1 domain và thực hiện quy trình kiểm tra Hybrid.
    """
    # 1. Tiền xử lý chuỗi
    domain_check = raw_input = raw_domain.lower().strip()
    if domain_check.endswith('.'): domain_check = domain_check[:-1] # Bỏ dấu chấm cuối DNS
    if domain_check.startswith("www."): domain_check = domain_check[4:]

    if not domain_check or len(domain_check) < 3: return

    print("-" * 60)
    print(f"[*] Dang phan tich: '{raw_input}' (Nguon: {src_ip})")

    # 2. LỚP 1: KIỂM TRA WHITELIST
    if domain_check in whitelist:
        print(f"    >>> [AN TOAN - WHITELIST] Website pho bien/uy tin.")
        return # Kết thúc ngay

    # 3. LỚP 2: PHÂN TÍCH HỌC MÁY
    start_time = time.time()
    try:
        # Trích xuất đặc trưng
        feat_dict = feature_utils.get_features_dict(domain_check)
        features_df = pd.DataFrame([feat_dict], columns=feature_utils.FEATURE_NAMES)
        
        # Dự đoán
        is_dga = int(model.predict(features_df)[0])
        proba = model.predict_proba(features_df)[0][1]
        process_time = (time.time() - start_time) * 1000

        # Hiển thị thông số kỹ thuật (Giải thích tại sao)
        print(f"    -> Thong so AI ({process_time:.2f}ms):")
        print(f"       Entropy: {feat_dict['entropy']:.2f} | Meaningful: {feat_dict['meaningful_ratio']*100:.0f}% | Bigram: {feat_dict['bigram_score']}")

        # Kết luận
        if is_dga == 1:
            print(f"    >>> [!!! CANH BAO] PHAT HIEN DGA MALWARE! <<<")
            print(f"        Do tin cay: {proba * 100:.2f}%")
            if proba > 0.8: print("        [!] KHUYEN NGHI: CHAN TRUY CAP NGAY LAP TUC.")
        else:
            print(f"    >>> [AN TOAN] Ten mien sach (Benign).")
            print(f"        Ty le DGA: {proba * 100:.2f}%")
            if proba > 0.4: print("        (Luu y: Ten mien hoi la, can than trong)")

    except Exception as e:
        print(f"    [!] Loi phan tich AI: {e}")

# --- HÀM XỬ LÝ FILE PCAP ---
def process_pcap_file(filepath, model, whitelist):
    if not os.path.exists(filepath):
        print(f"[!] Loi: Khong tim thay file '{filepath}'")
        return

    print(f"\n[*] Dang doc file PCAP: {filepath} ... (Vui long doi)")
    try:
        packets = rdpcap(filepath)
        print(f" -> Da doc {len(packets)} goi tin. Dang loc DNS Query...")
        
        dns_count = 0
        for pkt in packets:
            # Kiểm tra gói tin có lớp DNS và là Query (qr=0)
            if pkt.haslayer(DNS) and pkt.getlayer(DNS).qr == 0:
                try:
                    # Lấy tên miền query
                    qname = pkt.getlayer(DNSQR).qname.decode('utf-8')
                    
                    # Lấy IP nguồn (Máy nạn nhân đang hỏi DNS)
                    src_ip = "Unknown"
                    if pkt.haslayer(IP):
                        src_ip = pkt.getlayer(IP).src
                    
                    dns_count += 1
                    analyze_single_domain(qname, model, whitelist, src_ip)
                    
                except Exception as e:
                    continue
        
        if dns_count == 0:
            print("[!] Khong tim thay goi tin DNS nao trong file nay.")
        else:
            print(f"\n[DONE] Hoan tat phan tich {dns_count} truy van DNS.")

    except Exception as e:
        print(f"[!] Loi khi doc file PCAP: {e}")

# --- MAIN LOOP ---
def main():
    print_banner()
    model, whitelist = load_system()

    while True:
        print("\n" + "="*30)
        print(" CHON CHE DO HOAT DONG:")
        print(" 1. Nhap ten mien thu cong (Manual Input)")
        print(" 2. Quet file PCAP (Traffic Analysis)")
        print(" 3. Thoat (Exit)")
        print("="*30)
        
        choice = input("Lua chon cua ban (1/2/3): ").strip()

        if choice == '1':
            print("\n-- CHE DO NHAP TAY (Go 'menu' de quay lai) --")
            while True:
                domain = input("\n[Input] Nhap ten mien: ").strip()
                if domain.lower() == 'menu': break
                if domain.lower() == 'exit': sys.exit()
                if not domain: continue
                
                analyze_single_domain(domain, model, whitelist)

        elif choice == '2':
            print("\n-- CHE DO QUET FILE PCAP --")
            path = input("Nhap duong dan file .pcap (VD: traffic.pcap): ").strip()
            if path:
                process_pcap_file(path, model, whitelist)
            else:
                print("Ban chua nhap duong dan!")

        elif choice == '3':
            print("Tam biet!")
            break
        else:
            print("Lua chon khong hop le!")

if __name__ == "__main__":
    main()