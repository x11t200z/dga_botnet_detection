# 🛡️ DGA Domain Detection (XGBoost)

> **Phát hiện tên miền độc hại sinh ra bởi thuật toán (DGA - Domain Generation Algorithms) sử dụng học máy.**

Dự án này xây dựng một mô hình phát hiện các tên miền được sử dụng bởi các Botnet để kết nối với C&C Server. Mô hình trích xuất **9 đặc trưng** kết hợp với thuật toán **XGBoost** để phân loại tên miền là **benign** hay **DGA**.

**Độ chính xác hiện tại:** `88.42%`

---

## 📂 Cấu trúc dự án

```text
├── dataset/                     # Thư mục chứa dữ liệu thô (UMUDGA & Tranco)
├── 1_load.py                    # Bước 1: Load, làm sạch và cân bằng dữ liệu
├── 2_feature_extration.py       # Bước 2: Trích xuất 9 đặc trưng từ tên miền
├── 3_train_xgboost.py           # Bước 3: Huấn luyện model
├── 4_1_analyze.py               # Bước 4.1: Phân tích lỗi
├── 4_2_demo_model.py            # Bước 4.2: Demo kiểm tra tên miền thời gian thực
├── feature_utils.py             # Thư viện chứa các hàm tính toán đặc trưng
├── google_10k_words.txt         # Từ điển hỗ trợ tính 'meaningful ratio'
└── README.md                    # Tài liệu dự án
```
## Feature engineering
Mô hình không đọc chuỗi ký tự (text) mà phân tích dựa trên 9 đặc trưng ngôn ngữ học được tính toán trong feature_utils.py:

- Shannon entropy: Đo độ "hỗn loạn" của chuỗi ký tự (DGA thường có entropy cao).

- Meaningful word ratio: Tỷ lệ từ có nghĩa trong tên miền (dựa trên từ điển tiếng Anh 10k từ).

- Vowel ratio: Tỷ lệ nguyên âm (tên miền sạch thường dễ đọc -> nhiều nguyên âm hơn).

- Digit ratio: Tỷ lệ chữ số.

- Length: Độ dài tên miền.

- Max consonant sequence: Chuỗi phụ âm liên tiếp dài nhất.

- Hex character ratio: Tỷ lệ ký tự hexa (a-f, 0-9).

- Max identical characters: Ký tự lặp lại liên tiếp nhiều nhất.

- Bigram ccore: Điểm số dựa trên các cặp ký tự phổ biến trong tiếng Anh (th, er, on, ...).


## 🚀 Cài đặt & sử dụng

### 1. Prerequisites

Cài đặt các thư viện cần thiết:

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn tldextract joblib
```

### 2. Chuẩn bị dữ liệu

Dự án sử dụng 2 nguồn dữ liệu chính (Cần tải về và đặt vào folder `dataset`):

* **DGA:** [UMUDGA Dataset](https://www.google.com/search?q=https://umudga.inf.um.es/) (Đại học Murcia).
* **Benign:** [Tranco Top 1M](https://tranco-list.eu/).

### 3. Quy trình thực hiện

**B1: Xử lý dữ liệu thô**
Tổng hợp dữ liệu từ 50 họ malware và top website sạch, lọc bỏ nhiễu (CDN, IP address).

```bash
python 1_load.py
```

*Output: `dataset_full.csv*`*

**B2: Trích xuất đặc trưng**
Chuyển đổi tên miền thành các vector số học.

```bash
python 2_feature_extration.py
```

*Output: `dataset_training_ready.csv*`

**B3: Huấn luyện mô hình (Training)**
Chạy thuật toán XGBoost với **RandomizedSearchCV** để tìm tham số tối ưu.

```bash
python 3_train_xgboost.py
```

*Output: `dga_xgboost_model.pkl` và ma trận nhầm lẫn*

### 4. Đánh giá & Demo

**Phân tích sai số:**
Xem xét các trường hợp model nhận diện sai để tinh chỉnh feature.

```bash
python 4_1_analyze.py
```

**Chạy Demo (Live Test):**
Nhập tên miền bất kỳ để kiểm tra kết quả dự đoán.

```bash
python 4_2_demo_model.py
```

Ví dụ Demo:

```text
Nhap ten mien: google.com
-> [AN TOAN] Ten mien sach (Benign)

Nhap ten mien: aiybyuwbdyu.net
-> [CANH BAO] DGA MALWARE DETECTED!
```

## 📊 Kết quả
![()\[confusion_matrix_xgboost.png\]](confusion_matrix_xgboost.png)
## 🔮 Hướng phát triển

- Tích hợp vào các giải pháp như IDS, IPS, Firewall hoặc SIEM
---
