import pandas as pd
import math
import tldextract
from collections import Counter

# 1. Đọc file dữ liệu thô
print("Dang doc du lieu tu 'dataset_full.csv'...")
try:
    df = pd.read_csv('dataset_full.csv')
except FileNotFoundError:
    data = {
        'domain': ['google.com', 'facebook.com', 'ax7b9c1d.net', 'kjsfdkgj.ru'],
        'label': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    print("Khong tim thay file, dang dung du lieu mau de test.")

# Đảm bảo dữ liệu là string
df['domain'] = df['domain'].astype(str)

# --- BƯỚC QUAN TRỌNG: TÁCH TLD ---
print("Dang boc tach TLD (vi du: google.com -> google)...")

def extract_main_domain(domain):
    try:
        extracted = tldextract.extract(domain)
        return extracted.domain.lower()
    except:
        return domain.lower()

df['main_domain'] = df['domain'].apply(extract_main_domain)

# --- HÀM TÍNH ENTROPY ---
def calc_entropy(s):
    if not s:
        return 0
    p, lns = Counter(s), float(len(s))
    return -sum((count/lns) * math.log(count/lns, 2) for count in p.values())

# --- HÀM TRÍCH XUẤT ĐẶC TRƯNG ---
def extract_features(row):
    domain = str(row['main_domain']).lower()
    length = len(domain)

    # Nếu domain rỗng → trả về vector 0
    if length == 0:
        return [0, 0, 0, 0, 0]

    digits = sum(c.isdigit() for c in domain)
    digit_ratio = digits / length

    vowels = sum(c in 'aeiou' for c in domain)
    vowel_ratio = vowels / length

    consonants = sum(c.isalpha() and c not in 'aeiou' for c in domain)
    consonant_ratio = consonants / length

    # Ký tự dạng hex (a-f + 0-9) → DGA thường sinh kiểu này
    hex_chars = sum(c in '0123456789abcdef' for c in domain)
    hex_ratio = hex_chars / length

    return [
        length,
        digits,
        digit_ratio,
        vowel_ratio,
        hex_ratio
    ]

print("Dang tinh toan dac trung (Entropy, Length, Vowel Ratio...)...")

# 1. Tính entropy
df['entropy'] = df['main_domain'].apply(calc_entropy)

# 2. Tính các đặc trưng lexical khác
features_list = df.apply(extract_features, axis=1).tolist()
cols = ['length', 'digits', 'digit_ratio', 'vowel_ratio', 'hex_ratio']

df_features = pd.DataFrame(features_list, columns=cols)

# 3. Ghép lại
df_final = pd.concat([df, df_features], axis=1)

# --- XEM KẾT QUẢ ---
print("\n--- SO SANH KET QUA ---")
print(df_final[['domain', 'main_domain', 'entropy', 'length', 'label']].head(10))

# Lưu file để train
cols_to_keep = [
    'length', 'digits', 'digit_ratio',
    'vowel_ratio', 'hex_ratio', 'entropy', 'label'
]

df_final[cols_to_keep].to_csv('dataset_training_ready.csv', index=False)

print("\nDa luu file 'dataset_training_ready.csv'.")
print("San sang nap vao thuat toan XGBoost hoac RandomForest!")
