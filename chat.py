import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, jsonify
import torch
from huggingface_hub import login

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
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model: {base_model_id} on {device}...")  # Debugging print

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch_dtype,
    token=hf_token  # Use the token for authenticated access
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)

# Load the zero-shot classification model
medical_nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", token=hf_token)

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
    
    # Tokenization with attention mask
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)
    
    print(f"Model Inputs: {model_inputs}")  # Debugging print

    # Generation without sampling parameters (deterministic output)
    generated_ids = model.generate(
        model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None
    )

    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Generated Text: {generated_text}")  # Debugging print

    return jsonify({"response": generated_text, "is_medical": is_medical_query(prompt)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
