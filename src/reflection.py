from typing import List, Dict
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

class ReflectionEngine:
    def __init__(self, key_manager):
        self.key_manager = key_manager
        self.conversation_history = []
        self.setup_model()

    def setup_model(self):
        """Khởi tạo model để xử lý reflection"""
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.key_manager.get_api_key(),
            temperature=0.2,
        )

    def add_to_history(self, question: str, answer: str):
        """Thêm câu hỏi và câu trả lời vào lịch sử"""
        self.conversation_history.append({
            "question": question,
            "answer": answer
        })
        # Giữ lịch sử trong giới hạn 5 tương tác gần nhất
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    def generate_reflected_query(self, current_question: str) -> str:
        """Tạo câu query mới dựa trên lịch sử hội thoại"""
        if not self.conversation_history:
            return current_question

        prompt = f"""Dựa vào lịch sử hội thoại sau và câu hỏi hiện tại, hãy tạo một câu query mới 
        để tìm kiếm thông tin chính xác hơn. Chỉ trả về câu query mới, không cần giải thích.
        Nếu không có lịch sử, hãy giữ nguyên câu hỏi của người dùng.
        Lịch sử hội thoại:
        {self._format_history()}

        Câu hỏi hiện tại: {current_question}

        Query mới:"""

        try:
            self.model.google_api_key = self.key_manager.get_api_key()
            response = self.model.invoke(prompt)
            return str(response.content).strip()
        except Exception as e:
            print(f"Lỗi khi tạo reflected query: {str(e)}")
            return current_question

    def _format_history(self) -> str:
        """Format lịch sử hội thoại thành text"""
        formatted_history = []
        for i, interaction in enumerate(self.conversation_history, 1):
            formatted_history.append(f"Q{i}: {interaction['question']}")
            formatted_history.append(f"A{i}: {interaction['answer']}")
        return "\n".join(formatted_history) 