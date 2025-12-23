import pandas as pd
import numpy as np
import joblib
import tldextract
import math
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

print("--- KHOI DONG HE THONG PHAT HIEN DGA BOTNET ---")

# 1. Load model
try:
    print("Dang tai mo hinh AI...")
    model = joblib.load('dga_detection_model.pkl')
    print("-> Da tai mo hinh thanh cong!")
except FileNotFoundError:
    print("LOI: Khong tim thay file 'dga_detection_model.pkl'. Hay chay file train truoc!")
    exit()


# 2. Feature extraction
def extract_features_for_demo(domain):
    try:
        extracted = tldextract.extract(domain)
        main_domain = str(extracted.domain)
    except:
        main_domain = str(domain)

    if len(main_domain) == 0:
        return None

    length = len(main_domain)
    digits = sum(c.isdigit() for c in main_domain)
    digit_ratio = digits / length
    vowels = sum(1 for c in main_domain if c in 'aeiou')
    vowel_ratio = vowels / length
    hex_chars = sum(1 for c in main_domain if c in '0123456789abcdef')
    hex_ratio = hex_chars / length

    p, lns = Counter(main_domain), float(length)
    entropy = -sum((count/lns) * math.log((count/lns), 2) for count in p.values())

    features = pd.DataFrame([[length, digits, digit_ratio, vowel_ratio, hex_ratio, entropy]],
                            columns=['length', 'digits', 'digit_ratio', 'vowel_ratio', 'hex_ratio', 'entropy'])
    return features


# 3. Demo loop
print("\nHe thong san sang! Nhap 'exit' de thoat.")
print("-" * 50)

while True:
    user_input = input("\nNhap ten mien can kiem tra (VD: google.com): ").strip().lower()

    if user_input == 'exit':
        break
    if not user_input:
        continue

    features_df = extract_features_for_demo(user_input)

    if features_df is None:
        print("Loi: Ten mien khong hop le.")
        continue

    # Prediction
    prediction = int(model.predict(features_df)[0])
    proba = model.predict_proba(features_df)[0][1]   # Xác suất DGA

    if prediction == 1:
        print(f"KET QUA: '{user_input}' LA DGA Botnet!")
        print(f"Do nguy hiem: {proba * 100:.2f}%")
    else:
        print(f"KET QUA: '{user_input}' la ten mien sach.")
        print(f"Ty le DGA: {proba * 100:.2f}%")

    # Debug
    print(f"(Thong so: Entropy={features_df['entropy'].iloc[0]:.2f}, Length={features_df['length'].iloc[0]})")
