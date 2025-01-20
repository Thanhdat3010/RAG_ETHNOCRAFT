from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.load import dumps, loads
from typing import List
from langchain_core.documents import Document

class MultiQueryRetriever:
    def __init__(self, key_manager):
        self.key_manager = key_manager
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.key_manager.get_api_key(),
            temperature=0
        )
        
        # RAG-Fusion: Related - Hỗ trợ đa ngôn ngữ
        self.query_prompt = ChatPromptTemplate.from_template(
            """Bạn là trợ lý giúp tạo ra nhiều câu truy vấn tìm kiếm bằng tiếng Việt từ một câu hỏi đầu vào.
            Hãy tạo các câu truy vấn bằng tiếng Việt, không phân biệt ngôn ngữ đầu vào là gì.
            Giữ nguyên các tên riêng và thuật ngữ về dân tộc.

            Tạo các câu truy vấn liên quan đến: {question}

            Output (4 câu truy vấn bằng tiếng Việt):"""
        )

        # Tạo chain để generate queries và thêm câu hỏi gốc
        self.generate_queries = (
            self.query_prompt 
            | self.llm 
            | StrOutputParser() 
            | (lambda x: [x.split("\n")[0]] + x.split("\n"))  # Thêm câu hỏi gốc vào đầu list
        )

    def retrieve(self, question: str, retriever) -> List[Document]:
        """Thực hiện MultiQuery retrieval với hỗ trợ đa ngôn ngữ"""
        # Cập nhật key mới trước khi gọi API
        self.llm.google_api_key = self.key_manager.get_api_key()
        
        # Sử dụng chain operators
        retrieval_chain = (
            self.generate_queries
            | retriever.map()
        )
        
        # Thêm câu hỏi gốc vào đầu kết quả
        results = retrieval_chain.invoke({"question": question})
        
        # Làm phẳng danh sách kết quả và loại bỏ trùng lặp
        all_docs = []
        seen = set()
        
        for docs in results:
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_docs.append(doc)
                    
        return all_docs