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