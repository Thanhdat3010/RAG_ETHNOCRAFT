import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging
from typing import List, Dict, Union
import glob
import os
import sys

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sửa lại import paths
from src.text_processor import DocumentProcessor
from src.document_store import DocumentStore
from src.ranking import DocumentRanker
from src.question_classifier import QuestionClassifier
from src.rag_fusion import RAGFusionRetriever
from config.config import DATA_FOLDERS

class ChemistryExamGenerator:
    def __init__(self, key_manager, data_root="data"):
        logging.info("Khởi tạo Chemistry Exam Generator...")
        self.key_manager = key_manager
        self.data_root = data_root
        self.setup_model()
        self.setup_components()
        self.load_and_process_documents()

    def setup_model(self):
        """Khởi tạo model và embeddings"""
        api_key = self.key_manager.get_api_key()
        genai.configure(api_key=api_key)
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

    def setup_components(self):
        """Khởi tạo các components cho RAG"""
        self.doc_processor = DocumentProcessor()
        self.doc_store = DocumentStore(self.embeddings)
        self.ranker = DocumentRanker()
        self.rag_fusion = RAGFusionRetriever(self.key_manager)

    def load_and_process_documents(self):
        """Load và xử lý tài liệu cho RAG"""
        logging.info("Bắt đầu quá trình đọc tài liệu...")
        all_files = []
        
        # Duyệt qua tất cả thư mục con trong DATA_FOLDERS
        for folder in DATA_FOLDERS:
            folder_path = os.path.join(self.data_root, folder)
            if not os.path.exists(folder_path):
                logging.warning(f"Thư mục {folder_path} không tồn tại")
                continue
                
            # Thu thập files từ mỗi thư mục
            pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
            word_files = glob.glob(os.path.join(folder_path, "*.docx"))
            txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
            all_files.extend(pdf_files + word_files + txt_files)

        if not all_files:
            raise ValueError("Không tìm thấy tài liệu hỗ trợ trong các thư mục.")

        for file_path in all_files:
            current_hash = self.doc_store.calculate_file_hash(file_path)
            if file_path not in self.doc_store.file_hashes or self.doc_store.file_hashes[file_path] != current_hash:
                logging.info(f"Đang xử lý file mới/đã thay đổi: {file_path}")
                try:
                    texts = self.doc_processor.process_document(file_path)
                    logging.info(f"Đã tạo được {len(texts)} chunks từ file {file_path}")
                    self.doc_store.update_vectors(file_path, texts)
                except Exception as e:
                    logging.error(f"❌ Lỗi khi xử lý {file_path}: {str(e)}")

        self.vector_index = self.doc_store.get_retriever()
        logging.info("Hoàn tất quá trình đọc và xử lý tài liệu")

    def get_relevant_context(self, topic: str, grade: int) -> str:
        """Lấy ngữ cảnh liên quan từ tài liệu sử dụng RAG Fusion"""
        query = f"Hóa học lớp {grade} chủ đề {topic}"
        
        # Sử dụng RAG Fusion để lấy documents có score
        doc_scores = self.rag_fusion.retrieve(query, self.vector_index)
        
        # Chỉ lấy documents (bỏ scores) và trích xuất page_content
        retrieved_docs = []
        for doc_score in doc_scores:
            if isinstance(doc_score, tuple):
                doc, score = doc_score
                retrieved_docs.append(doc)
            else:
                retrieved_docs.append(doc_score)
        
        logging.info(f"Tìm thấy {len(retrieved_docs)} tài liệu liên quan")
        
        # Rerank documents
        logging.info("Bắt đầu rerank tài liệu...")
        reranked_docs = self.ranker.rerank_documents(query, retrieved_docs)
        logging.info("Hoàn thành rerank tài liệu")
        
        # Xử lý và làm sạch ngữ cảnh
        contexts = [self.clean_context(doc.page_content) for doc in reranked_docs]
        return "\n\n".join(contexts)

    def clean_context(self, text: str) -> str:
        """Làm sạch và format lại ngữ cảnh"""
        text = text.replace('\\n', ' ')
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = ' '.join(line.split())
            if line and not line.isspace():
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)

    def generate_exam(self, grade: int, topic: str, difficulty: str,
                     total_questions: int,
                     multiple_choice: int = 0,
                     true_false: int = 0,
                     short_answer: int = 0) -> str:
        """Tạo đề thi dựa trên các tham số đầu vào và ngữ cảnh từ RAG"""
        try:
            logging.info(f"Bắt đầu tạo đề thi: Lớp {grade}, Chủ đề: {topic}, Độ khó: {difficulty}")
            
            # Cập nhật API key
            self.model.google_api_key = self.key_manager.get_api_key()
            
            # Lấy ngữ cảnh liên quan từ tài liệu
            context = self.get_relevant_context(topic, grade)
            logging.info("Đã lấy được ngữ cảnh liên quan từ tài liệu")
            
            # Tạo prompt với ngữ cảnh
            prompt = self.create_exam_prompt(
                grade, topic, difficulty,
                total_questions, multiple_choice,
                true_false, short_answer,
                context
            )
            
            # Gọi model để tạo đề thi
            response = self.model.invoke(prompt)
            
            # Lấy text từ response
            if hasattr(response, 'content'):
                exam_text = response.content
            else:
                exam_text = str(response)
                
            if not exam_text or exam_text.strip() == "":
                raise ValueError("Không nhận được phản hồi từ model")
                
            logging.info("Tạo đề thi thành công")
            return exam_text
            
        except Exception as e:
            error_msg = f"Lỗi khi tạo đề thi: {str(e)}"
            logging.error(error_msg)
            return f"Đã xảy ra lỗi khi tạo đề thi: {error_msg}"

    def create_exam_prompt(self, grade: int, topic: str, difficulty: str,
                         total_questions: int,
                         multiple_choice: int = 0,
                         true_false: int = 0,
                         short_answer: int = 0,
                         context: str = "") -> str:
        """Tạo prompt template cho việc sinh đề thi với ngữ cảnh từ RAG"""
        
        return f"""Bạn là một chuyên gia trong việc tạo đề thi hóa học.
        
        Dựa trên ngữ cảnh sau về chủ đề:
        {context}
        
        Hãy tạo bộ câu hỏi hóa học lớp {grade} với chủ đề {topic} và độ khó {difficulty}.
        Tổng số câu hỏi cần tạo là {total_questions} câu, bao gồm:
        {f'- {multiple_choice} câu hỏi trắc nghiệm với 4 lựa chọn' if multiple_choice > 0 else ''}
        {f'- {true_false} câu hỏi đúng/sai với 4 phát biểu' if true_false > 0 else ''}
        {f'- {short_answer} câu hỏi điền đáp án ngắn' if short_answer > 0 else ''}

        Lưu ý quan trọng: nếu chủ đề không liên quan tới hóa thì tự tạo ngẫu nhiên một chủ đề liên quan tới hóa học.
        
        Yêu cầu QUAN TRỌNG về định dạng:
        1. TUYỆT ĐỐI KHÔNG sử dụng bất kỳ thẻ HTML nào
        2. KHÔNG sử dụng các ký tự đặc biệt hay định dạng HTML
        3. Chỉ sử dụng văn bản thuần túy (plain text)
        4. Với các công thức hóa học:
         - Viết chỉ số dưới bằng ký tự Unicode trực tiếp (ví dụ: H₂O, CO₂)
         - Sử dụng ký tự → cho mũi tên phản ứng
         - Sử dụng dấu ⇌ cho phản ứng thuận nghịch
        5. Với các đơn vị đo:
         - Viết m³ thay vì m3
         - Viết cm³ thay vì cm3
         - Viết độ C thay vì °C
        6. Với các số mũ và chỉ số:
         - Sử dụng ký tự Unicode trực tiếp (ví dụ: x², x₁, x₂)

        Các yêu cầu về nội dung:
        1. Tạo đủ số lượng câu hỏi theo yêu cầu
        2. Các câu hỏi không được giống nhau, các đáp án trong cùng một câu không được giống nhau
        3. Câu hỏi được đặt bằng tiếng Việt
        4. Đảm bảo các công thức hóa học có chỉ số dưới dạng subscript
        5. Đảm bảo các câu hỏi chỉ liên quan đến môn hóa học
        6. Tham khảo ngữ cảnh được cung cấp để tạo câu hỏi phù hợp và chính xác
        
        Quy tắc về danh pháp và thuật ngữ:
        1. **Giữ nguyên danh pháp hóa học và các thuật ngữ chuyên ngành giống trong ngữ cảnh**:
        - Không dịch hoặc thay đổi danh pháp hóa học (bao gồm tên tiếng Anh hoặc IUPAC)
        - Nếu ngữ cảnh bằng tiếng Anh, hãy dịch sang tiếng Việt nhưng giữ nguyên danh pháp hóa học giống trong ngữ cảnh(tiếng anh)
        2. **TUYỆT ĐỐI KHÔNG DỊCH tên các chất hoá học**:
        - Giữ nguyên 100% tên gọi của các chất hoá học như trong ngữ cảnh
        - Ví dụ: nếu ngữ cảnh viết "phosphoric acid" thì PHẢI giữ nguyên là "phosphoric acid", KHÔNG được dịch thành "axit photphoric"

        Yêu cầu cho từng loại câu hỏi:
        - Trắc nghiệm: 4 lựa chọn, 1 đáp án đúng và giải thích chi tiết
        - Đúng/sai: 4 phát biểu liên kết, có câu dẫn, phát biểu cuối khó nhất
        + Lưu ý: số câu hỏi đúng sai khi người dùng nhập là 4 thì phải tạo đủ 4 câu hỏi và mỗi câu 4 phát biểu chứ không phải 1 câu hỏi.
        - Trả lời ngắn: chỉ tạo câu hỏi tính toán và có đáp án ngắn gọn (không có chữ), không dùng dạng toán đốt cháy.
        Lưu ý: không reset Số thứ tự qua các phần.
        
        Định dạng câu trả lời:
        PHẦN I. CÂU HỎI TRẮC NGHIỆM
        [Số thứ tự]. [Nội dung câu hỏi]
        A. [Đáp án A]
        B. [Đáp án B]
        C. [Đáp án C]
        D. [Đáp án D]
        Đáp án đúng: [Chữ cái đáp án]
        Giải thích: [Giải thích chi tiết]

        PHẦN II. CÂU HỎI ĐÚNG/SAI
        [Số thứ tự]. [Câu dẫn]
        1. [Phát biểu 1] - [Đúng/Sai]
        2. [Phát biểu 2] - [Đúng/Sai]
        3. [Phát biểu 3] - [Đúng/Sai]
        4. [Phát biểu 4] - [Đúng/Sai]

        PHẦN III. CÂU HỎI TÍNH TOÁN
        [Số thứ tự]. [Nội dung bài toán]
        Đáp án: [Kết quả tính toán]
        """

    def validate_input(self, grade: int, topic: str, difficulty: str,
                      total_questions: int,
                      multiple_choice: int,
                      true_false: int,
                      short_answer: int) -> bool:
        """Kiểm tra tính hợp lệ của input"""
        if not (8 <= grade <= 12):
            raise ValueError("Lớp học phải từ 8-12")
            
        if not topic or len(topic.strip()) == 0:
            raise ValueError("Chủ đề không được để trống")
            
        if difficulty not in ["Dễ", "Trung bình", "Khó"]:
            raise ValueError("Độ khó không hợp lệ")
            
        if total_questions != (multiple_choice + true_false + short_answer):
            raise ValueError("Tổng số câu hỏi không khớp với số lượng các loại")
            
        return True 

if __name__ == "__main__":
    import sys
    import os
    
    # Thêm thư mục gốc vào PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from config.config import key_manager, DATA_ROOT
    
    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Khởi tạo và test
    try:
        # Sử dụng key_manager từ config
        exam_generator = ChemistryExamGenerator(key_manager, DATA_ROOT)
        
        # Nhập thông tin đề thi
        grade = int(input("Nhập lớp (8-12): "))
        topic = input("Nhập chủ đề: ")
        difficulty = input("Nhập độ khó (Dễ/Trung bình/Khó): ")
        multiple_choice = int(input("Số câu trắc nghiệm: "))
        true_false = int(input("Số câu đúng/sai: "))
        short_answer = int(input("Số câu tự luận: "))
        total_questions = multiple_choice + true_false + short_answer
        
        # Validate và tạo đề
        exam_generator.validate_input(
            grade=grade,
            topic=topic,
            difficulty=difficulty,
            total_questions=total_questions,
            multiple_choice=multiple_choice,
            true_false=true_false,
            short_answer=short_answer
        )
        
        print("\nĐang tạo đề thi...")
        exam = exam_generator.generate_exam(
            grade=grade,
            topic=topic,
            difficulty=difficulty,
            total_questions=total_questions,
            multiple_choice=multiple_choice,
            true_false=true_false,
            short_answer=short_answer
        )
        
        print("\n=== ĐỀ THI ĐÃ TẠO ===\n")
        print(exam)
        print("\n===================\n")
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")