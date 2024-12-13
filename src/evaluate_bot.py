import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bot import ChemGenieBot
from config.config import APIKeyManager, DATA_FOLDERS, GOOGLE_API_KEYS
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import pandas as pd
from datetime import datetime

def create_gemini_evaluator():
    api_key = GOOGLE_API_KEYS[0]
    genai.configure(api_key=api_key)
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=api_key,
        temperature=0.1,
    )

def evaluate_chembot(bot: ChemGenieBot):
    evaluator = create_gemini_evaluator()
    results = []
    
    for i, (question, ground_truth) in enumerate(zip(eval_questions, eval_answers)):
        # Lấy câu trả lời từ bot
        bot_answer = bot.ask_question(question)
        
        # Tạo prompt đánh giá
        eval_prompt = f"""Hãy đánh giá câu trả lời của chatbot dựa trên các tiêu chí sau, cho điểm từ 0-1:

        Câu hỏi: {question}
        Câu trả lời mẫu: {ground_truth}
        Câu trả lời của bot: {bot_answer}

        Tiêu chí đánh giá:
        1. Độ liên quan (Answer Relevancy): Câu trả lời có trả lời đúng câu hỏi không?
        2. Độ trung thực (Faithfulness): Thông tin trong câu trả lời có chính xác và đáng tin cậy không?
        3. Độ chính xác ngữ cảnh (Context Precision): Ngữ cảnh được sử dụng có phù hợp không?
        4. Độ bao phủ ngữ cảnh (Context Recall): Câu trả lời có sử dụng đầy đủ thông tin từ ngữ cảnh không?

        Hãy trả về điểm số theo format chính xác:
        faithfulness: [điểm 0-1]
        answer_relevancy: [điểm 0-1]
        context_precision: [điểm 0-1]
        context_recall: [điểm 0-1]"""

        # Lấy đánh giá từ Gemini
        eval_result = evaluator.invoke(eval_prompt)
        eval_text = eval_result.content

        # Parse điểm số
        scores = {
            'faithfulness': 0,
            'answer_relevancy': 0,
            'context_precision': 0,
            'context_recall': 0
        }
        
        try:
            for line in eval_text.split('\n'):
                for metric in scores.keys():
                    if f"{metric}:" in line.lower():
                        scores[metric] = float(line.split(':')[1].strip())
        except Exception as e:
            print(f"Lỗi khi parse kết quả đánh giá cho câu hỏi {i+1}: {e}")

        # Tạo dictionary cho một hàng trong DataFrame
        row = {
            'question': question,
            'answer': bot_answer,
            'ground_truth': ground_truth,
            'faithfulness': scores['faithfulness'],
            'answer_relevancy': scores['answer_relevancy'],
            'context_precision': scores['context_precision'],
            'context_recall': scores['context_recall']
        }
        results.append(row)
        
        # In tiến độ
        print(f"Đã đánh giá {i+1}/{len(eval_questions)} câu hỏi")

    # Tạo DataFrame
    df = pd.DataFrame(results)
    
    # Tính điểm trung bình
    averages = df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean()
    
    return df, averages

# Định nghĩa câu hỏi và câu trả lời mẫu về hóa học
eval_questions = [
    "Ester là gì?",
    "Tính chất vật lý của ester là gì?",
    "Ứng dụng của ester trong thực tế?",
    "Polymer là gì?",
    "Phân loại polymer theo nguồn gốc?",
    "Kể tên một số ứng dụng của polymer tổng hợp trong đời sống?",
    "Carbohydrate là gì và phân loại các loại đường?",
    "Vai trò của carbohydrate trong cơ thể?",
    "Amine là gì? Phân loại amine?",
    "Tính chất hóa học cơ bản của amine?"
]

eval_answers = [
    "Ester là hợp chất hữu cơ được tạo thành từ phản ứng giữa alcohol và acid carboxylic, có công thức cấu tạo R-COO-R' trong đó R và R' là gốc hydrocarbon.",
    "Ester thường là chất lỏng ở điều kiện thường, có mùi thơm đặc trưng của hoa quả, tan tốt trong dung môi hữu cơ nhưng ít tan trong nước.",
    "Ester được ứng dụng rộng rãi trong công nghiệp thực phẩm làm hương liệu, trong mỹ phẩm, dược phẩm và làm dung môi hữu cơ.",
    "Polymer là những phân tử lớn (đại phân tử) được tạo thành từ nhiều đơn vị cấu trúc nhỏ (monomer) liên kết với nhau thông qua liên kết hóa học.",
    "Polymer được phân loại thành: 1) Polymer thiên nhiên: có sẵn trong tự nhiên như cellulose, tinh bột, protein. 2) Polymer bán tổng hợp: được điều chế từ polymer thiên nhiên. 3) Polymer tổng hợp: được tổng hợp hoàn toàn từ các monomer như PE, PVC, PS.",
    "Polymer tổng hợp có nhiều ứng dụng quan trọng như: làm vật liệu đóng gói (túi nilon, hộp nhựa), sản xuất quần áo (polyester, nylon), vật liệu xây dựng (ống nhựa PVC), linh kiện điện tử, và nhiều sản phẩm tiêu dùng khác.",
    "Carbohydrate là những hợp chất hữu cơ gồm C, H, O với tỉ lệ H:O = 2:1. Phân loại: 1) Monosaccharide (đường đơn): glucose, fructose; 2) Disaccharide (đường đôi): sucrose, lactose; 3) Polysaccharide (đường đa): tinh bột, cellulose, glycogen.",
    "Carbohydrate có vai trò: 1) Cung cấp năng lượng chính cho cơ thể (1g glucose sinh ra 4kcal); 2) Tham gia cấu tạo tế bào; 3) Dự trữ năng lượng dưới dạng glycogen trong gan và cơ; 4) Tham gia quá trình trao đổi chất.",
    "Amine là dẫn xuất của amoniac (NH3), trong đó một hoặc nhiều nguyên tử H được thay thế bằng gốc hydrocarbon. Phân loại: 1) Amine bậc một (RNH2); 2) Amine bậc hai (R2NH); 3) Amine bậc ba (R3N).",
    "Tính chất hóa học của amine: 1) Tính base mạnh hơn NH3; 2) Tác dụng với acid tạo muối; 3) Tham gia phản ứng thế với halogen; 4) Tham gia phản ứng alkyl hóa; 5) Phản ứng với HNO2 tạo khí N2."
]

if __name__ == "__main__":
    # Khởi tạo bot và chạy đánh giá
    key_manager = APIKeyManager(GOOGLE_API_KEYS)
    data_root = "data"
    bot = ChemGenieBot(key_manager, data_root)
    
    # Chạy đánh giá
    print("Bắt đầu đánh giá...")
    results_df, averages = evaluate_chembot(bot)
    
    # Tạo tên file với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = f"evaluation_results_{timestamp}.xlsx"
    
    # Lưu kết quả vào Excel
    with pd.ExcelWriter(excel_file) as writer:
        results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        pd.DataFrame(averages).to_excel(writer, sheet_name='Averages')
    
    print(f"\nKết quả đã được lưu vào file: {excel_file}")
    print("\n=== Điểm trung bình ===")
    for metric, score in averages.items():
        print(f"{metric}: {score:.3f}") 