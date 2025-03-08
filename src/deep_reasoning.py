from typing import List, Dict
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import time

class DeepReasoning:
    def __init__(self, key_manager):
        self.key_manager = key_manager
        self.setup_model()
        
    def setup_model(self):
        api_key = self.key_manager.get_api_key()
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=2048,
        )

    def deep_think(self, question: str, context: str) -> Dict:
        """
        Thực hiện quá trình suy nghĩ sâu dựa trên context có sẵn
        """
        # Kiểm tra context trước
        if not context or context.isspace():
            return {
                "thoughts": [{
                    "step": "Kiểm tra thông tin",
                    "thought": "🔍 Đang kiểm tra nguồn thông tin...",
                    "content": "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
                }],
                "final_answer": "Tôi không có đủ thông tin để trả lời câu hỏi này."
            }

        # Bước 0: Kiểm tra độ liên quan của context
        relevance_prompt = f"""Dựa trên context sau:
        {context}
        
        Hãy phân tích xem context có chứa đủ thông tin để trả lời câu hỏi: "{question}" không?
        
        CHÚ Ý:
        - Chỉ trả lời YES nếu context chứa thông tin trực tiếp hoặc có thể suy luận hợp lý từ thông tin có sẵn
        - Trả lời NO nếu context không chứa đủ thông tin liên quan
        - Chỉ trả lời YES hoặc NO"""

        has_relevant_info = self._get_llm_response(relevance_prompt).strip().upper()
        
        if has_relevant_info != "YES":
            return {
                "thoughts": [{
                    "step": "Kiểm tra độ liên quan",
                    "thought": "🔍 Đang đánh giá thông tin...",
                    "content": "Context không chứa đủ thông tin liên quan để trả lời câu hỏi này."
                }],
                "final_answer": "Tôi không có đủ thông tin để trả lời câu hỏi này."
            }

        thoughts = []
        
        # Bước 1: Phân tích tổng hợp
        analysis_prompt = f"""Dựa trên context sau:
        {context}
        
        1. Phân tích câu hỏi: "{question}"
        2. Xác định các điểm chính liên quan
        3. Tìm ra mối liên hệ giữa các thông tin
        4. Đưa ra các nhận định quan trọng

        Hãy suy luận một cách logic và khách quan.
        
        CHÚ Ý: Chỉ sử dụng thông tin từ context được cung cấp."""

        analysis = self._get_llm_response(analysis_prompt)
        thoughts.append({
            "step": "Phân tích tổng hợp",
            "thought": "🔍 Đang phân tích và kết nối thông tin...",
            "content": analysis
        })

        # Bước 2: Đưa ra kết luận
        conclusion_prompt = f"""Dựa trên phân tích trên:
        {analysis}
        
        Hãy đưa ra câu trả lời cho câu hỏi: "{question}"
        
        YÊU CẦU:
        - Trả lời ngắn gọn, súc tích
        - Đưa ra kết luận rõ ràng, chắc chắn
        - Tập trung vào những điểm chính đã phân tích"""

        final_answer = self._get_llm_response(conclusion_prompt)
        
        return {
            "thoughts": thoughts,
            "final_answer": final_answer
        }

    def _get_llm_response(self, prompt: str) -> str:
        try:
            self.model.google_api_key = self.key_manager.get_api_key()
            response = self.model.invoke(prompt)
            # Thêm delay ngắn để người dùng theo dõi được quá trình suy nghĩ
            time.sleep(1.5)
            return response.content
        except Exception as e:
            logging.error(f"Lỗi khi gọi LLM: {str(e)}")
            return "Đã xảy ra lỗi trong quá trình suy luận" 