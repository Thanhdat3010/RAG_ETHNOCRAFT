from langchain_google_genai import ChatGoogleGenerativeAI

class Reflection:
    def __init__(self, key_manager):
        self.key_manager = key_manager
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.key_manager.get_api_key(),
            temperature=0.1
        )
        
    def _format_chat_history(self, chat_history):
        """Format lịch sử chat thành chuỗi văn bản."""
        formatted_messages = []
        for message in chat_history:
            role = message.get('role', '')
            content = message.get('content', '')
            formatted_messages.append(f"{role}: {content}")
        return "\n".join(formatted_messages)

    def __call__(self, chat_history, max_history=2):
        """
        Phân tích lịch sử chat và chuyển đổi câu hỏi thành dạng độc lập.
        
        Args:
            chat_history (list): Lịch sử chat
            max_history (int): Số lượng tin nhắn tối đa xem xét
        
        Returns:
            str: Câu hỏi được chuyển đổi
        """
        # Cập nhật API key mới
        self.model.google_api_key = self.key_manager.get_api_key()
        
        # Giới hạn lịch sử chat
        if len(chat_history) > max_history:
            chat_history = chat_history[-max_history:]

        # Format lịch sử chat
        history_text = self._format_chat_history(chat_history)

        # Tạo prompt
        prompt = f"""Dựa vào lịch sử chat và câu hỏi mới nhất của người dùng (có thể tham chiếu đến ngữ cảnh trong lịch sử chat), 
        hãy chuyển đổi thành một câu hỏi độc lập bằng tiếng Việt có thể hiểu được mà không cần lịch sử chat. 
        Quang trọng: Nếu câu hỏi đã độc lập, không cần chuyển đổi.
        Ví dụ: "Ester là gì" thì không được chuyển đổi thành "Este là gì?"
        KHÔNG trả lời câu hỏi, chỉ chuyển đổi nếu cần và giữ nguyên nếu không cần.
        **Giữ nguyên danh pháp hóa học và các thuật ngữ chuyên ngành giống câu hỏi**:
            - Không dịch hoặc thay đổi danh pháp hóa học (bao gồm tên tiếng Anh hoặc IUPAC).
            - Nếu ngữ cảnh bằng tiếng Anh, hãy dịch sang tiếng Việt nhưng giữ nguyên danh pháp hóa học giống trong ngữ cảnh(tiếng anh).
            - **TUYỆT ĐỐI KHÔNG DỊCH tên các chất hoá học**:
            - Giữ nguyên 100% tên gọi của các chất hoá học như trong ngữ cảnh
            - Ví dụ: nếu ngữ cảnh viết "phosphoric acid" thì PHẢI giữ nguyên là "phosphoric acid", KHÔNG được dịch thành "axit photphoric"
        Lịch sử chat:
        {history_text}
        """
        
        response = self.model.invoke(prompt)
        return response.content 