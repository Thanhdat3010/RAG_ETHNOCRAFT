import os
import random
from typing import List
import logging

class APIKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.call_count = 0
    
    def get_api_key(self) -> str:
        self.call_count += 1
        key = random.choice(self.api_keys)
        # Chỉ hiện 8 ký tự đầu của key để bảo mật
        masked_key = key[:8] + "..." 
        logging.info(f"API Call #{self.call_count} - Using key: {masked_key}")
        return key

# List các API keys
GOOGLE_API_KEYS = [
    "AIzaSyCdm8-0p8SQREhPDicyywkDeDbMVtMmhNo",
    "AIzaSyBc1fHj2tGSwmVraM39ZXzFjvy_qubMct8",
    "AIzaSyAwokgee5qqhev3eZaQ3dhqXN23UrLHpNo",
    "AIzaSyBNIoGntqsjY0ElbfrJmlBHYLep0QbYOAo",

]

# Tạo instance của API Key Manager
key_manager = APIKeyManager(GOOGLE_API_KEYS)

# Thay đổi cấu trúc thư mục
DATA_ROOT = "data"  # Thư mục gốc chứa dữ liệu
DATA_FOLDERS = [    # Các thư mục con chứa tài liệu
    "DANTOCTHIEUSO",
    "Dân tộc-Thông tin bổ sung",
    "LUẬN ÁN DÂN TỘC",
    "QTH B ca sang T5",
    "Dữ liệu mới",
    "Học tiếng dân tộc Khmer",
    "Bài báo Dân tộc thiểu số",
    "THI CUỐI KÌ",
    # Thêm các thư mục khác tùy nhu cầu
]
VECTOR_STORE_PATH = "vector_store"
FILE_HASH_PATH = "file_hashes/file_hashes.pkl"