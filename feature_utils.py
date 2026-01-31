import math
import re
from collections import Counter
import tldextract
import os

# --- CẤU HÌNH LOAD TỪ ĐIỂN ---
WORDLIST_FILE = 'google_10k_words.txt'
COMMON_WORDS_SET = set()

def load_google_10k():
    """Load dictionary into a set for fast lookup."""
    global COMMON_WORDS_SET
    if not COMMON_WORDS_SET and os.path.exists(WORDLIST_FILE):
        with open(WORDLIST_FILE, 'r', encoding='utf-8') as f:
            COMMON_WORDS_SET = {line.strip().lower() for line in f if line.strip()}
    return COMMON_WORDS_SET

# Load dictionary at import
load_google_10k()

# Top Bigrams vẫn giữ nguyên
COMMON_BIGRAMS = ['in', 'er', 'th', 'on', 'an', 'en', 'co', 're', 'or', 'st']

def extract_main_domain(domain_str):
    try:
        domain_str = str(domain_str).lower()
        extracted = tldextract.extract(domain_str)
        # If subdomain is long, it might be DGA part
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
    Optimized version: Check substrings against the dictionary set.
    """
    clean_domain = re.sub(r"[^a-z]", "", domain)
    if not clean_domain: return 0
    
    n = len(clean_domain)
    # Dynamic programming or simple greedy match could be used.
    # Here is a greedy approach for performance: find longest valid word starting at each position.
    
    # However, to be closer to "ratio", we can count valid char coverage.
    # Simple optimization: Iterate through all substrings and check if valid.
    # But that's O(N^2). Since N is small (<255), it's fast.
    
    # Better approach for "ratio":
    # Mark characters that belong to any meaningful word.
    
    is_meaningful = [False] * n
    
    if not COMMON_WORDS_SET:
        return 0

    # Max length of words in dictionary could be an optimization, but for 10k works it's fine.
    # Most english words are < 20 chars.
    max_word_len = 20 
    
    for i in range(n):
        for j in range(i, min(i + max_word_len, n)):
            sub = clean_domain[i : j+1]
            if sub in COMMON_WORDS_SET:
                # Mark range as meaningful
                for k in range(i, j+1):
                    is_meaningful[k] = True
                    
    return sum(is_meaningful) / n

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