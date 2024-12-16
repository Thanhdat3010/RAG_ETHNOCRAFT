import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model và tokenizer từ Hugging Face
model_name = "FiveC/finetune-llama2-7b"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Sử dụng float16 để giảm memory
    device_map="auto"  # Tự động map vào GPU nếu có
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Tạo pipeline để generate text
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    truncation=True,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

# Test model
prompt = "Hóa học phân tích là gì"
result = pipe(f"{prompt}")
print(result[0]['generated_text'])

# Giải phóng bộ nhớ khi không cần dùng nữa
del model, pipe
torch.cuda.empty_cache()