import os
import time
from dotenv import load_dotenv # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, jsonify # type: ignore
import torch # type: ignore
from huggingface_hub import login # type: ignore

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face API token from environment
hf_token = os.getenv("HUGGINGFACE_TOKEN")

if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN is not set in the .env file")

# Login to Hugging Face Hub
login(token=hf_token)

app = Flask(__name__)

# Load the chatbot model
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cpu"  # Force CPU mode

# Optimize CPU performance
torch.set_num_threads(4)  # Allow multithreading
torch.backends.mkldnn.enabled = True  # Enable MKL optimizations

print(f"Loading model: {base_model_id} on {device}...")  # Debugging print

# Load model without bitsandbytes quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map="cpu",  # Use simple device map
    low_cpu_mem_usage=True,  # Reduce memory footprint
    trust_remote_code=True  # Ensure full model download
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load zero-shot classification model
medical_nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def is_medical_query(query):
    candidate_labels = ["medical", "non-medical"]
    result = medical_nlp(query, candidate_labels)
    return result["labels"][0] == "medical" and result["scores"][0] > 0.85

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "")

    if not prompt:
        return jsonify({"error": "Message is required"}), 400

    normal_response = [
        {"role": "system", "content": "You are an NLP assistant designed to generate a complete and useful response."},
        {"role": "user", "content": f"## Input:\n{prompt}\n\n## Response:"}
    ]

    text = tokenizer.apply_chat_template(normal_response, tokenize=False, add_generation_prompt=True)

    # Tokenization with optimized parameters
    start_time = time.time()
    model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    print(f"Tokenization Time: {time.time() - start_time:.2f}s")

    # Generation with optimized parameters
    start_time = time.time()
    generated_ids = model.generate(
        model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=128,  # Reduced for faster response
        do_sample=True,  # Enable sampling for faster inference
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )
    print(f"Generation Time: {time.time() - start_time:.2f}s")

    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({"response": generated_text, "is_medical": is_medical_query(prompt)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
