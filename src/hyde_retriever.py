from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from .hybrid_retriever import HybridRetriever

class HyDERetriever:
    def __init__(self, key_manager, vector_store):
        self.key_manager = key_manager
        self.hybrid_retriever = HybridRetriever(vector_store)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.key_manager.get_api_key(),
            temperature=0.1
        )
        
        self.hyde_prompt = ChatPromptTemplate.from_template(
            """Bạn là một chuyên gia hóa học. Hãy viết một đoạn văn ngắn trả lời câu hỏi sau.
            Đoạn văn cần ngắn gọn, súc tích và chứa các thuật ngữ chuyên ngành hóa học liên quan.
            
            Câu hỏi: {question}
            
            Trả lời bằng tiếng Việt và tiếng Anh, mỗi ngôn ngữ một đoạn."""
        )

    def retrieve(self, query: str, **kwargs):
        # Tạo hypothetical document
        hyde_response = self.llm.invoke(
            self.hyde_prompt.format(question=query)
        ).content
        
        # Kết hợp query gốc với hypothetical document
        augmented_query = f"""
        Câu hỏi: {query}
        
        Thông tin liên quan:
        {hyde_response}
        """
        
        # Sử dụng HybridRetriever với query đã tăng cường
        results = self.hybrid_retriever.get_relevant_documents(augmented_query, **kwargs)
        # Chỉ trả về documents (bỏ scores)
        return [doc for doc, _ in results] 