from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import logging
import json
import time
from src.bot import EthnoAI
from config.config import key_manager, DATA_ROOT
import warnings

# Bỏ qua tất cả warnings từ PyPDF
warnings.filterwarnings('ignore')

# Hoặc cụ thể hơn cho warnings liên quan đến Xref
warnings.filterwarnings('ignore', message='.*Xref table invalid.*')

# Tùy chọn: Set logging level cho pypdf lên ERROR
logging.getLogger('pypdf').setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

bot = EthnoAI(key_manager, DATA_ROOT)

@app.route('/chat', methods=['POST'])
def chat():
    logging.info("Nhận được yêu cầu chat mới")
    try:
        message = request.form.get('message')
        if not message:
            return jsonify({'error': 'Không có tin nhắn được cung cấp'}), 400
        
        response = bot.ask_question(message)
        return jsonify({'response': response})
    
    except Exception as e:
        logging.error(f"Lỗi server: {str(e)}")
        return jsonify({'error': 'Lỗi server nội bộ'}), 500

@app.route('/deep-think', methods=['POST'])
def deep_think():
    logging.info("Nhận được yêu cầu deep thinking")
    try:
        data = request.get_json()
        message = data.get('message')
        if not message:
            return jsonify({'error': 'Không có tin nhắn được cung cấp'}), 400

        def generate_thinking_steps():
            try:
                # Bước 0: Khởi động suy nghĩ
                yield f"data: {json.dumps({'type': 'thinking', 'content': '🧠 Bắt đầu quá trình suy nghĩ sâu...\n\nTôi sẽ:\n1. Tìm kiếm thông tin liên quan\n2. Phân tích tổng hợp\n3. Đưa ra kết luận'}, ensure_ascii=False)}\n\n"
                time.sleep(1.5)

                # Bước 1: Tìm kiếm thông tin
                yield f"data: {json.dumps({'type': 'thinking', 'content': '🔍 Đang tìm kiếm trong cơ sở dữ liệu...'}, ensure_ascii=False)}\n\n"
                retrieved_docs = bot.query_retriever.retrieve(message, bot.vector_index)
                reranked_docs = bot.ranker.rerank_documents(message, retrieved_docs)
                contexts = [bot.clean_context(doc.page_content) for doc in reranked_docs]
                context = "\n\n".join(contexts)
                time.sleep(1.5)

                # Bước 2: Thực hiện suy luận sâu
                result = bot.deep_reasoning.deep_think(message, context)
                
                # Hiển thị kết quả phân tích
                for thought in result["thoughts"]:
                    thought_content = f"{thought['thought']}\n\n{thought['content']}"
                    yield f"data: {json.dumps({'type': 'thinking', 'content': thought_content}, ensure_ascii=False)}\n\n"
                    time.sleep(1.5)

                # Bước cuối: Kết luận
                yield f"data: {json.dumps({'type': 'thinking', 'content': '✨ Đang tổng hợp câu trả lời cuối cùng...'}, ensure_ascii=False)}\n\n"
                time.sleep(1)

                # Trả về câu trả lời cuối cùng
                final_response = f"📝 Kết luận:\n\n{result['final_answer']}"
                yield f"data: {json.dumps({'type': 'final', 'content': final_response}, ensure_ascii=False)}\n\n"

            except Exception as e:
                error_msg = f"❌ Lỗi trong quá trình suy luận: {str(e)}"
                logging.error(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate_thinking_steps()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Content-Type': 'text/event-stream',
                'Access-Control-Allow-Origin': '*'
            }
        )

    except Exception as e:
        logging.error(f"Lỗi server khi xử lý deep thinking: {str(e)}")
        return jsonify({'error': 'Lỗi server nội bộ'}), 500

if __name__ == "__main__":
    logging.info("Đang khởi động ứng dụng ChemGenie Bot...")
    app.run(debug=True)