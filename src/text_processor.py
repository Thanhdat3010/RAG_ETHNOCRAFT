from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
import logging
import os
from config.config import DATA_FOLDERS
import re

# File này để xử lý tài liệu
class DocumentProcessor:
    @staticmethod
    def load_document(file_path: str):
        # Kiểm tra xem file có nằm trong thư mục hợp lệ không
        if not any(folder in file_path.split(os.sep) for folder in DATA_FOLDERS):
            raise ValueError(f"File không nằm trong thư mục được cấu hình: {file_path}")
            
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            return PyPDFLoader(file_path)
        elif file_extension in ['.doc', '.docx']:
            return UnstructuredWordDocumentLoader(file_path)
        elif file_extension == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        raise ValueError(f"Unsupported file type: {file_extension}")

    @staticmethod
    def clean_chunk(chunk: str) -> str:
        # Xử lý các pattern phổ biến
        replacements = [
            # Xử lý các từ khóa về hình ảnh/bảng biểu
            (r'\[(?:image|img|figure|table).*?\]', ' '),  # [image], [Image 2.1], [Table 1], etc.
            (r'\b(?:image|figure|table)\s*\d*\b', ' '),   # image 1, figure 2.1, etc.
            
            # Xử lý dấu chấm và ký tự đặc biệt
            (r'\.{2,}', '. '),        # Nhiều dấu chấm thành một dấu chấm
            (r'\s*[_*]{2,}\s*', ' '), # Loại bỏ ____ hoặc **** 
            
            # Xử lý khoảng trắng và xuống dòng
            (r'\s+', ' '),            # Gộp nhiều khoảng trắng thành một
            (r'\n+', ' '),            # Gộp nhiều xuống dòng thành khoảng trắng
        ]
        
        # Áp dụng các pattern
        for pattern, replacement in replacements:
            chunk = re.sub(pattern, replacement, chunk, flags=re.IGNORECASE)
        
        # Xử lý dấu câu
        chunk = re.sub(r'\s+([.,!?])', r'\1', chunk)  # Xóa space trước dấu câu
        chunk = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', chunk)  # Thêm space sau dấu câu
        
        return chunk.strip()

    @staticmethod
    def process_document(file_path: str):
        try:
            loader = DocumentProcessor.load_document(file_path)
            pages = loader.load_and_split()
            context = "\n\n".join(str(p.page_content) for p in pages)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            texts = text_splitter.split_text(context)
            
            # Clean và lọc chunks
            cleaned_chunks = []
            for chunk in texts:
                cleaned = DocumentProcessor.clean_chunk(chunk)
                if len(cleaned.strip()) > 50:  # chỉ giữ lại chunks đủ dài
                    cleaned_chunks.append(cleaned)
            
            return cleaned_chunks
        except Exception as e:
            logging.error(f"Error processing document {file_path}: {str(e)}")
            raise