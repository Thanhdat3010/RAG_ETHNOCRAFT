from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
# File này để phân loại câu hỏi có phải là giao tiếp thông thường hay câu hỏi chuyên môn cần RAG
class QuestionClassifier:
    def __init__(self, key_manager):
        self.key_manager = key_manager
        self.classifier = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.key_manager.get_api_key(),
            temperature=0
        )
        self.load_prompts()
        self.conversation_history = []  # Thêm conversation history
    
    def load_prompts(self):
        """Load prompts from config file"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'prompts.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                self.classification_prompt = prompts.get('classification_prompt', self.default_classification_prompt())
                self.conversation_prompt = prompts.get('conversation_prompt', self.default_conversation_prompt())
        except FileNotFoundError:
            self.classification_prompt = self.default_classification_prompt()
            self.conversation_prompt = self.default_conversation_prompt()
    
    def default_classification_prompt(self):
        return """Hãy phân loại câu hỏi dựa trên ngữ cảnh hội thoại sau:

        LỊCH SỬ HỘI THOẠI:
        {history}

        CÂU HỎI HIỆN TẠI: {question}

        Phân loại thành một trong ba loại sau:
        1. "CHAT" - Nếu là câu hỏi giao tiếp thông thường (ví dụ: chào hỏi, hỏi thăm, giới thiệu, cảm xúc...)
        2. "FOLLOW_UP" - Nếu là câu hỏi tiếp nối hoặc liên quan đến các câu hỏi trước (ví dụ: "còn gì nữa không", "kể thêm đi", "chi tiết hơn được không")
        3. "KNOWLEDGE" - Nếu là câu hỏi mới cần tra cứu thông tin hoặc kiến thức
        4. Dù được hỏi thông tin một cách thân mật thì vẫn phải trả về "KNOWLEDGE"
        Chỉ trả lời "CHAT", "FOLLOW_UP" hoặc "KNOWLEDGE", không giải thích thêm.
        """
    
    def default_conversation_prompt(self):
        return """Bạn là một trợ lý BVAI(Bách VIệt AI) thân thiện. Hãy trả lời câu hỏi sau một cách thân thiện:
        {question}
        Câu trả lời mẫu:
        Chào bạn! Mình là trợ lý BVAI, mình sẽ giải đáp mọi câu hỏi về dân tộc Việt Nam ta, tôi sẽ đồng hành với bạn trong quá trình tìm hiểu văn hóa dân tộc.
        """
    
    def _format_history(self):
        """Format lịch sử hội thoại"""
        if not self.conversation_history:
            return "Chưa có lịch sử hội thoại."
        
        formatted = []
        for i, (q, a) in enumerate(self.conversation_history, 1):
            formatted.extend([f"Q{i}: {q}", f"A{i}: {a}"])
        return "\n".join(formatted)

    def add_to_history(self, question, answer):
        """Thêm câu hỏi và câu trả lời vào lịch sử"""
        self.conversation_history.append((question, answer))
        if len(self.conversation_history) > 5:  # Giữ 5 tương tác gần nhất
            self.conversation_history.pop(0)

    def is_conversational(self, question):
        """Xác định loại câu hỏi dựa trên ngữ cảnh"""
        self.classifier.google_api_key = self.key_manager.get_api_key()
        response = self.classifier.invoke(
            self.classification_prompt.format(
                history=self._format_history(),
                question=question
            )
        )
        question_type = response.content.strip().upper()
        
        # Nếu là câu hỏi kiến thức mới hoặc follow-up, trả về False để xử lý bằng RAG
        return question_type == "CHAT"
    
    def get_conversation_response(self, question):
        """Get response for conversational questions"""
        self.classifier.google_api_key = self.key_manager.get_api_key()
        return self.classifier.invoke(
            self.conversation_prompt.format(question=question)
        ).content 