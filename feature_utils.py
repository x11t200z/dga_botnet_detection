import math
import re
from collections import Counter
import tldextract
import os

# --- CẤU HÌNH LOAD TỪ ĐIỂN ---
WORDLIST_FILE = 'google_10k_words.txt'
COMMON_WORDS_SET = set()

# Load từ điển khi import module
if os.path.exists(WORDLIST_FILE):
    with open(WORDLIST_FILE, 'r', encoding='utf-8') as f:
        # Đọc từng dòng và đưa vào set
        COMMON_WORDS_SET = {line.strip().lower() for line in f}
else:
    # Fallback nếu quên chưa tải file (Giữ lại danh sách cũ làm dự phòng)
    COMMON_WORDS_SET = {
        'online', 'store', 'site', 'web', 'group', 'company', 'cloud', 'server', 'mail', 
        'app', 'play', 'game', 'net', 'org', 'com', 'edu', 'gov', 'info', 'biz',
        'news', 'blog', 'tech', 'shop', 'bank', 'card', 'secure', 'login'
    }

# Top Bigrams vẫn giữ nguyên
COMMON_BIGRAMS = ['in', 'er', 'th', 'on', 'an', 'en', 'co', 're', 'or', 'st']

def extract_main_domain(domain_str):
    try:
        domain_str = str(domain_str).lower()
        extracted = tldextract.extract(domain_str)
        if len(extracted.subdomain) > 3: 
             return f"{extracted.subdomain}.{extracted.domain}"
        return extracted.domain
    except: 
        return str(domain_str).lower()

def calc_entropy(s):
    if not s: return 0
    p, lns = Counter(s), float(len(s))
    return -sum((count/lns) * math.log(count/lns, 2) for count in p.values())

def meaningful_word_ratio_simple(domain):
    """
    Thuat toan Greedy Match voi tu dien 10,000 tu.
    """
    clean_domain = re.sub(r"[^a-z]", "", domain)
    original_len = len(clean_domain)
    if original_len == 0: return 0
    
    # Sắp xếp từ dài trước ngắn sau để ưu tiên từ dài (VD: 'notification' > 'not')
    # Lưu ý: Việc sort 10k từ mỗi lần chạy hàm sẽ chậm -> Nên sort 1 lần ở ngoài nếu muốn tối ưu cực đại
    # Nhưng với Python hiện đại, việc này vẫn rất nhanh.
    sorted_words = sorted(list(COMMON_WORDS_SET), key=len, reverse=True)
    
    # Tối ưu: Chỉ lấy các từ CÓ XUẤT HIỆN trong domain để loop (giảm số vòng lặp)
    potential_words = [w for w in sorted_words if w in clean_domain]
    
    found_len = 0
    temp_domain = clean_domain
    
    for word in potential_words:
        if word in temp_domain:
            found_len += len(word)
            temp_domain = temp_domain.replace(word, "", 1)
            
    return min(found_len / original_len, 1.0)

def hex_char_ratio(domain):
    if not domain: return 0
    hex_chars = set('0123456789abcdef')
    count = sum(c in hex_chars for c in domain)
    return count / len(domain)

def max_consecutive_identical(domain):
    if not domain: return 0
    max_len = 1
    current_len = 1
    for i in range(1, len(domain)):
        if domain[i] == domain[i-1]:
            current_len += 1
            max_len = max(max_len, current_len)
        else:
            current_len = 1
    return max_len

def common_bigram_count(domain):
    if len(domain) < 2: return 0
    count = 0
    for bigram in COMMON_BIGRAMS:
        if bigram in domain:
            count += 1
    return count

# --- HÀM CHÍNH ---
def get_features_dict(raw_domain):
    main_domain = extract_main_domain(raw_domain)
    length = len(main_domain)
    
    if length == 0: return {k: 0 for k in FEATURE_NAMES}

    digits = sum(c.isdigit() for c in main_domain)
    digit_ratio = digits / length
    
    vowel_count = sum(c in 'aeiou' for c in main_domain)
    vowel_ratio = vowel_count / length
    
    consonants = "bcdfghjklmnpqrstvwxyz0123456789"
    current_len = 0
    max_consonant_len = 0
    for char in main_domain:
        if char in consonants:
            current_len += 1
            max_consonant_len = max(max_consonant_len, current_len)
        else:
            current_len = 0

    entropy = calc_entropy(main_domain)
    meaningful = meaningful_word_ratio_simple(main_domain)
    hex_ratio = hex_char_ratio(main_domain)
    max_identical = max_consecutive_identical(main_domain)
    bigram_score = common_bigram_count(main_domain)

    return {
        'length': length,
        'digit_ratio': digit_ratio,
        'vowel_ratio': vowel_ratio,
        'max_consonant_len': max_consonant_len,
        'entropy': entropy,
        'meaningful_ratio': meaningful,
        'hex_ratio': hex_ratio,
        'max_identical': max_identical,
        'bigram_score': bigram_score
    }

FEATURE_NAMES = [
    'length', 'digit_ratio', 'vowel_ratio', 
    'max_consonant_len', 'entropy', 'meaningful_ratio',
    'hex_ratio', 'max_identical', 'bigram_score'
]