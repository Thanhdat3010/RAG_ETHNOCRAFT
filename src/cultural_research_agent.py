import google.generativeai as genai
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import logging  # Nhập logging
from langchain_core.prompts import PromptTemplate  # Nhập PromptTemplate

GOOGLE_API_KEY = "AIzaSyBFQPRDlpG9bXs3-_To8j8M2X9FnEDLe4E"  # Thay thế bằng API Key của bạn
genai.configure(api_key=GOOGLE_API_KEY)

class CulturalResearchAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logging.info("CulturalResearchAgent đã được khởi tạo.")  # Log khi khởi tạo
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
    def rewrite_query(self, original_query: str) -> str:
        """Sử dụng LLM để viết lại câu hỏi cho tìm kiếm web."""
        prompt = f"Viết lại câu hỏi sau để tìm kiếm thông tin: {original_query}. ví dụ với query là: viết báo cáo về dân tộc kinh. Thì phải viết lại là thông tin về dân tộc kinh. Lưu ý chỉ viết lại 1 query, không đề cập gì khác"
        response = self.model.generate_content(prompt)
        return response.text.strip()  # Trả về câu hỏi đã được viết lại

    def search_web(self, query, max_results=5):
        """Tìm kiếm thông tin trên web sử dụng DuckDuckGo Search API."""
        logging.info(f"Tìm kiếm web với truy vấn: {query}")  # Log truy vấn tìm kiếm
        
        # Viết lại câu hỏi trước khi tìm kiếm
        rewritten_query = self.rewrite_query(query)
        logging.info(f"Câu hỏi đã được viết lại: {rewritten_query}")  # Log câu hỏi đã viết lại
        
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(rewritten_query, max_results=max_results)]
            logging.info(f"Tìm thấy {len(results)} kết quả.")  # Log số lượng kết quả
            for result in results:  # Log từng kết quả tìm kiếm
                logging.info(f"Kết quả tìm kiếm: {result['body']} - URL: {result['href']}")
            return results if results else []

    def evaluate_source_reliability(self, url):
        """Đánh giá độ tin cậy của nguồn tài liệu."""
        logging.info(f"Đánh giá độ tin cậy của nguồn: {url}")  # Log URL đang đánh giá
        try:
            response = requests.get(url)
            response.raise_for_status()  # Kiểm tra lỗi HTTP
            soup = BeautifulSoup(response.content, 'html.parser')

            if "gov.vn" in url:
                return "Nguồn chính phủ, độ tin cậy cao."
            elif "edu.vn" in url:
                return "Nguồn giáo dục, độ tin cậy khá cao."
            return "Nguồn không rõ ràng, cần kiểm tra thêm."
        except requests.exceptions.RequestException as e:
            logging.error(f"Lỗi truy cập URL: {e}")  # Log lỗi nếu có
            return f"Lỗi truy cập URL: {e}"

    def filter_sensitive_content(self, text):
        """Lọc bỏ nội dung nhạy cảm, phản động."""
        prompt = f"Phân tích và lọc bỏ nội dung nhạy cảm, phản động (nếu có) trong đoạn văn bản sau:\n\n{text}"
        response = self.model.generate_content(prompt)
        return response.text

    def generate_report(self, query, rag_results=None):
        """Tạo báo cáo nghiên cứu từ kết quả tìm kiếm và RAG."""
        logging.info(f"Tạo báo cáo cho truy vấn: {query}")  # Log truy vấn tạo báo cáo
        search_results = self.search_web(query)
        report_content = ""
        context = ""  # Khởi tạo biến context

        # Tạo context từ kết quả RAG nếu có
        if rag_results:
            # Giả sử rag_results là danh sách các chuỗi văn bản
            contexts = [self.clean_context(doc.page_content) for doc in rag_results]
            context = "\n\n".join(contexts)
            logging.info(f"Ngữ cảnh từ RAG: {context}")  # Log ngữ cảnh để kiểm tra

        # Thêm kết quả tìm kiếm từ web
        for result in search_results:
            url = result['href']
            reliability = self.evaluate_source_reliability(url)
            content = result['body']
            filtered_content = self.filter_sensitive_content(content)

            report_content += f"**Nguồn:** [{url}]({url})\n"
            report_content += f"**Độ tin cậy:** {reliability}\n"
            report_content += f"**Nội dung:** {filtered_content}\n\n"

        if report_content or context:
            logging.info("Báo cáo đã được tạo thành công.")  # Log khi báo cáo được tạo
            
            # Định nghĩa template
            template = """Ta có thông tin từ các nguồn sau:

            {report_content}

            Thông tin từ BVAI
            {context}
            Viết một bài báo cáo chuyên nghiệp theo cấu trúc chuẩn gồm các phần sau:

            + Tiêu đề: Ngắn gọn, súc tích, phản ánh nội dung chính của báo cáo.
            + Mở đầu: Giới thiệu ngắn gọn về chủ đề báo cáo, mục tiêu và phạm vi nghiên cứu.
            + Nội dung chính: Trình bày thông tin theo các mục rõ ràng, có thể bao gồm:
            + Cơ sở lý thuyết hoặc bối cảnh
            + Phương pháp thực hiện
            + Kết quả thu được
            + Phân tích và đánh giá
            + Kết luận và đề xuất: Tổng hợp các ý chính, đưa ra kết luận và đề xuất (nếu có).
            + Tài liệu tham khảo: Liệt kê các nguồn tài liệu sử dụng(quan trọng), ghi độ tin cậy bên cạnh.
            Viết bằng văn phong trang trọng, mạch lạc, có logic và dễ hiểu.
            """
            
            # Tạo PromptTemplate
            prompt_template = PromptTemplate.from_template(template)
            prompt = prompt_template.format(report_content=report_content, context=context)
            response = self.model.generate_content(prompt)
            return response.text
        
        logging.warning("Không tìm thấy kết quả để tạo báo cáo.")  # Log nếu không có kết quả
        return "Không tìm thấy kết quả."

    def run(self):
        """Chạy AI Agent."""
        query = input("Nhập chủ đề nghiên cứu: ")
        report = self.generate_report(query)
        print(report)
        methods = self.suggest_research_methods(query)
        print(methods)

if __name__ == "__main__":
    agent = CulturalResearchAgent()
    agent.run() 