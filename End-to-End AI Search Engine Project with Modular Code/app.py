from flask import Flask, request, jsonify, render_template
import faiss
import ollama
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS

# ========== Initialize Flask ==========
app = Flask(__name__, template_folder="templates")

# ========== Fix Torch Issues ==========
try:
    _ = torch.__version__
except Exception as e:
    print(f"Torch Initialization Failed: {e}")
    exit()

# ========== Load Sentence Transformer Model ==========
try:
    bert_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
except Exception as e:
    print(f"Model Load Error: {e}")
    exit()

# ========== Initialize FAISS Index ==========
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
product_data = []

# ========== DeepSeek API Configuration ==========
DEEPSEEK_70B_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_70B_API_KEY = "sk-00d362e53f074a0e841473716799428f"  # **Replace with actual API key**

# ========== Function: Encode Text ==========
def encode_text(text):
    return bert_model.encode([text])[0]

# ========== Function: Add Product ==========
@app.route("/add_product", methods=["POST"])
def add_product():
    data = request.json
    name, description = data.get("name"), data.get("description")

    if not name or not description:
        return jsonify({"error": "Missing name or description"}), 400

    vector = encode_text(description)
    index.add(np.array([vector], dtype=np.float32))
    product_data.append({"name": name, "description": description})

    return jsonify({"message": f"✅ Product '{name}' added!"})

# ========== Function: Search Products ==========
@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Local search
    local_results = []
    if index.ntotal > 0:
        query_vector = encode_text(query)
        distances, indices = index.search(np.array([query_vector], dtype=np.float32), 5)
        local_results = [product_data[idx] for idx in indices[0] if idx < len(product_data)]

    # External search
    external_results = []
    try:
        with DDGS() as ddgs:
            external_results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        external_results = [{"title": "Error fetching results", "href": "#", "body": str(e)}]

    # AI response (Ollama)
    ai_response = generate_response(query, local_results)

    # DeepSeek-70B analysis
    deepseek_70b_response = analyze_with_deepseek_70b(query, local_results, external_results)

    return jsonify({
        "local_results": local_results,
        "external_results": external_results,
        "ai_response": ai_response,
        "deepseek_70b_response": deepseek_70b_response
    })

# ========== Function: AI Response using DeepSeek-1.3B (Ollama) ==========
def generate_response(query, results):
    results_text = "\n".join([f"- {r['name']}: {r['description']}" for r in results])
    prompt = f"You are an AI assistant recommending products.\n\nUser query: {query}\n\nMatching products:\n{results_text}\n\nRecommend the best product."

    try:
        response = ollama.chat(model="deepseek-coder:1.3b", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"❌ AI Response Error: {str(e)}"

# ========== Function: DeepSeek-70B API Analysis ==========
def analyze_with_deepseek_70b(query, local_results, external_results):
    local_text = "\n".join([f"- {r['name']}: {r['description']}" for r in local_results])
    external_text = "\n".join([f"- {r['title']} ({r['href']})" for r in external_results])

    prompt = f"User searched for: {query}\n\nLocal results:\n{local_text}\n\nExternal results:\n{external_text}\n\nAnalyze the best products."

    headers = {"Authorization": f"Bearer {DEEPSEEK_70B_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-70b", "messages": [{"role": "user", "content": prompt}]}

    try:
        response = requests.post(DEEPSEEK_70B_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except requests.exceptions.RequestException as e:
        return f"❌ API Error: {str(e)}"

# ========== Serve HTML ==========
@app.route("/")
def home():
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True)
