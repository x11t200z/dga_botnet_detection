import requests
import os

# URL của bộ từ điển Google 10,000 từ tiếng Anh
URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
SAVE_PATH = "google_10k_words.txt"

print(f"Dang tai tu dien tu GitHub... ({URL})")

try:
    response = requests.get(URL)
    if response.status_code == 200:
        # Lọc bỏ các từ quá ngắn (dưới 3 ký tự) để tránh nhiễu
        words = [w.strip() for w in response.text.splitlines() if len(w.strip()) > 3]
        
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(words))
            
        print(f"[OK] Da luu {len(words)} tu vung vao file '{SAVE_PATH}'")
        print("Hay cap nhat feature_utils.py ngay!")
    else:
        print("Loi tai file!")
except Exception as e:
    print(f"Loi: {e}")