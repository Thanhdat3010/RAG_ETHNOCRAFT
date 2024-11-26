import os
import pickle
import hashlib
from langchain_community.vectorstores import Chroma
import logging
from config.config import VECTOR_STORE_PATH, FILE_HASH_PATH, DATA_FOLDERS
from .hybrid_retriever import HybridRetriever  # Thêm import
# File này để lưu trữ và quản lý các tài liệu và vectors
class DocumentStore:
    def __init__(self, embeddings):
        self.vector_store_path = VECTOR_STORE_PATH
        self.file_hash_path = FILE_HASH_PATH
        # Tạo thư mục file_hashes nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.file_hash_path), exist_ok=True)
        self.embeddings = embeddings
        self.file_hashes = self.load_file_hashes()
        self.vector_store = self.initialize_vector_store()
        # Thêm dòng này để tự động cleanup khi khởi tạo
        self.cleanup_deleted_files()

    def load_file_hashes(self):
        if os.path.exists(self.file_hash_path):
            with open(self.file_hash_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_file_hashes(self):
        with open(self.file_hash_path, 'wb') as f:
            pickle.dump(self.file_hashes, f)

    def calculate_file_hash(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def initialize_vector_store(self):
        if os.path.exists(self.vector_store_path):
            return Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
        return Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.vector_store_path
        )

    def get_retriever(self, k=5):
        """Return hybrid retriever thay vì Chroma retriever mặc định"""
        return HybridRetriever(
            vector_store=self.vector_store,
            alpha=0.5,  # Có thể điều chỉnh
            k=k
        )

    def update_vectors(self, file_path: str, texts: list):
        try:
            logging.info(f"Bắt đầu cập nhật vectors cho file: {file_path}")
            # Delete old vectors if they exist
            if file_path in self.file_hashes:
                logging.info(f"Xóa vectors cũ của file: {file_path}")
                self.vector_store.delete(where={"source": file_path})
            
            # Add new vectors with additional metadata
            logging.info(f"Thêm {len(texts)} chunks mới vào vector store")
            # Xác định subject từ đường dẫn file
            subject = next((folder for folder in DATA_FOLDERS 
                          if folder in file_path.split(os.sep)), "unknown")
            
            metadatas = [{
                "source": file_path,
                "subject": subject
            } for _ in texts]
            
            self.vector_store.add_texts(texts, metadatas=metadatas)
            
            # Update hash
            self.file_hashes[file_path] = self.calculate_file_hash(file_path)
            self.save_file_hashes()
            
            self.vector_store.persist()
            logging.info(f"Hoàn thành cập nhật vectors cho file: {file_path}")
            return True
        except Exception as e:
            logging.error(f"❌ Lỗi khi cập nhật vectors cho {file_path}: {str(e)}")
            return False

    def cleanup_deleted_files(self):
        """Xóa vectors của những file không còn tồn tại."""
        deleted_files = []
        for file_path in self.file_hashes.keys():
            if not os.path.exists(file_path):
                logging.info(f"Phát hiện file đã xóa: {file_path}")
                self.vector_store.delete(where={"source": file_path})
                deleted_files.append(file_path)
        
        # Cập nhật file_hashes
        for file_path in deleted_files:
            del self.file_hashes[file_path]
        
        if deleted_files:
            self.save_file_hashes()
            self.vector_store.persist()
            logging.info(f"Đã xóa vectors của {len(deleted_files)} files không còn tồn tại")