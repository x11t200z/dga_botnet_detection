import pandas as pd
import joblib
import warnings
import os
import sys

try:
    import feature_utils
except ImportError:
    print("LOI: Khong tim thay file 'feature_utils.py'.")
    print("Vui long tao file nay truoc khi chay demo.")
    sys.exit()

warnings.filterwarnings("ignore")

model_filename = 'dga_rf_model.pkl'
print(f"[*] Dang tai mo hinh '{model_filename}'...")

if not os.path.exists(model_filename):
    print(f"[!] LOI: Khong tim thay file '{model_filename}'.")
    print("    Vui long chay file '3_train_random_forest.py' de huan luyen lai mo hinh voi 6 dac trung moi!")
    sys.exit()

try:
    model = joblib.load(model_filename)
    print(" -> [OK] Da tai model thanh cong!")
    
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(feature_utils.FEATURE_NAMES):
        print(f"\n[CANH BAO] Model hien tai duoc train voi {model.n_features_in_} dac trung.")
        print(f"           Nhung code demo dang dung {len(feature_utils.FEATURE_NAMES)} dac trung.")
        print("           -> VUI LONG CHAY LAI FILE TRAIN DE CAP NHAT MODEL!")
except Exception as e:
    print(f"[!] Loi khi tai model: {e}")
    sys.exit()

print("\nHe thong san sang! Nhap 'exit' de thoat.")
print("-" * 60)

while True:
    try:
        user_input = input("\nNhap ten mien (VD: google.com, xq99z.net): ").strip().lower()
    except KeyboardInterrupt:
        break

    if user_input == 'exit':
        print("Tam biet!")
        break
    if not user_input:
        continue

    feat_dict = feature_utils.get_features_dict(user_input)
    
    main_domain = feature_utils.extract_main_domain(user_input)

    features_df = pd.DataFrame([feat_dict], columns=feature_utils.FEATURE_NAMES)

    print(f"\n-> Phan tich domain: '{main_domain}'")
    print("   [Thong so ky thuat]:")
    print(f"   1. Do dai (Length):        {feat_dict['length']}")
    print(f"   2. Ty le so (Digits):      {feat_dict['digit_ratio']*100:.1f}%")
    print(f"   3. Nguyen am (Vowels):     {feat_dict['vowel_ratio']*100:.1f}%  (Thap => Kho doc)")
    print(f"   4. Phu am lien tiep (Max): {feat_dict['max_consonant_len']}     (Cao => Bat thuong)")
    print(f"   5. Entropy (Do hon loan):  {feat_dict['entropy']:.2f}    (>3.5 thuong la DGA)")
    print(f"   6. Co nghia (Meaningful):  {feat_dict['meaningful_ratio']*100:.1f}%  (Cao => Sach)")

    try:
        prediction = int(model.predict(features_df)[0])
        proba = model.predict_proba(features_df)[0][1] # Xác suất là DGA

        print("\n   [KET QUA CUOI CUNG]:")
        if prediction == 1:
            print(f"   >>> [CANH BAO] DGA MALWARE DETECTED! <<<")
        else:
            print(f"   >>> [AN TOAN] Ten mien sach (Benign). <<<")
            
    except Exception as e:
        print(f"\n[!] Loi khi du doan: {e}")