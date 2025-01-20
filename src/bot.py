import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging
import glob
import os
from .text_processor import DocumentProcessor
from .document_store import DocumentStore
from .ranking import DocumentRanker
from .question_classifier import QuestionClassifier
from .multi_query import MultiQueryRetriever
from config.config import DATA_FOLDERS
from .reflection import ReflectionEngine

class EthnoAI:
    def __init__(self, key_manager, data_root):
        logging.info("Đang khởi tạo ChemGenieBot...")
        self.key_manager = key_manager
        self.data_root = data_root
        self.setup_model()
        self.setup_components()
        self.load_and_process_documents()
        self.setup_qa_chain()
        self.reflection_engine = ReflectionEngine(key_manager)

    def setup_model(self):
        api_key = self.key_manager.get_api_key()
        genai.configure(api_key=api_key)
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.2,
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

    def setup_components(self):
        self.doc_processor = DocumentProcessor()
        self.doc_store = DocumentStore(self.embeddings)
        self.ranker = DocumentRanker()
        self.question_classifier = QuestionClassifier(self.key_manager)
        self.query_retriever = MultiQueryRetriever(self.key_manager)

    def load_and_process_documents(self):
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

    def setup_qa_chain(self):
        template = """Bạn là một chatbot AI chuyên về các dân tộc thiểu số Việt Nam. Hãy trả lời một cách chính xác, ngắn gọn và dễ hiểu.

        Dựa trên các thông tin được cung cấp dưới đây:
        {context}

        Hãy trả lời câu hỏi sau: {question}

        Hướng dẫn:
        - Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp
        - Nếu thông tin trong ngữ cảnh không đủ để trả lời, hãy nói "Tôi không có đủ thông tin để trả lời câu hỏi này"
        - Không được tự tạo thêm thông tin
        - Trả lời bằng giọng điệu thân thiện, tôn trọng
        - Ngữ cảnh(tài liệu) đó cứ coi là kiến thức của bạn, không được đề cập là "dựa vào tài liệu cung cấp" mà chỉ trả lời câu hỏi của người dùng một cách vui vẻ và thân thiện

        Câu trả lời:"""
        
        prompt = PromptTemplate.from_template(template)
        
        def get_response(inputs):
            self.model.google_api_key = self.key_manager.get_api_key()
            return self.model.invoke(inputs)
        
        self.qa_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()} 
            | prompt 
            | get_response
            | StrOutputParser()
        )

    def clean_context(self, text):
        """Làm sạch và format lại ngữ cảnh"""
        # Thay thế \n bằng khoảng trắng trước khi tách dòng
        text = text.replace('\\n', ' ')
        
        # Tách các dòng và xử lý tiếp
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Loại bỏ khoảng trắng thừa và các ký tự đặc biệt
            line = ' '.join(line.split())
            if line and not line.isspace():
                cleaned_lines.append(line)
        
        # Gộp các dòng thành đoạn văn
        paragraphs = []
        current_paragraph = []
        
        for line in cleaned_lines:
            if line.strip() in ['○', '•']:  # Markers for new paragraph
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
            current_paragraph.append(line)
            
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            
        return '\n\n'.join(paragraphs)

    def ask_question(self, question):
        logging.info(f"Nhận được câu hỏi: {question}")
        try:
            # Cập nhật API key mới cho model trước khi xử lý
            self.model.google_api_key = self.key_manager.get_api_key()
            
            # Phân loại câu hỏi
            if self.question_classifier.is_conversational(question):
                logging.info("Phát hiện câu hỏi giao tiếp, sử dụng LLM trực tiếp")
                response = self.question_classifier.get_conversation_response(question)
                self.reflection_engine.add_to_history(question, response)
                return response
            
            # Sử dụng reflection để tạo câu query mới
            reflected_question = self.reflection_engine.generate_reflected_query(question)
            logging.info(f"Câu hỏi sau khi reflection: {reflected_question}")

            # Xử lý câu hỏi chuyên môn bằng RAG với câu query đã được reflection
            logging.info("Đang tìm kiếm tài liệu liên quan...")
            retrieved_docs = self.query_retriever.retrieve(reflected_question, self.vector_index)
            logging.info(f"Tìm thấy {len(retrieved_docs)} tài liệu liên quan")
            
            logging.info("Bắt đầu rerank tài liệu...")
            reranked_docs = self.ranker.rerank_documents(reflected_question, retrieved_docs)
            logging.info("Hoàn thành rerank tài liệu")
            
            # Xử lý và làm sạch ngữ cảnh
            contexts = [self.clean_context(doc.page_content) for doc in reranked_docs]
            context = "\n\n".join(contexts)
            
            logging.info("Đang tạo câu trả lời...")
            prompt = {
                "context": context,
                "question": reflected_question
            }
            print("=== PROMPT ===")
            print(prompt)
            print("=============")
            
            answer = self.qa_chain.invoke({"context": context, "question": reflected_question})
            
            if not answer or answer.strip() == "":
                logging.warning("Không thể tạo câu trả lời")
                return "Xin lỗi, tôi không thể tạo câu trả lời. Vui lòng thử lại."
            
            if answer and answer.strip() != "":
                self.reflection_engine.add_to_history(question, answer)
                
            logging.info("Đã tạo xong câu trả lời")
            return answer
            
        except Exception as e:
            logging.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
            return "Đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại."