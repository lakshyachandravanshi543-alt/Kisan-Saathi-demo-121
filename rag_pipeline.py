import os
import requests
import json

def get_answer(question, language="English"):
    """
    Direct REST API function to get an answer. 
    Bypasses Langchain and gRPC completely to prevent Streamlit Cloud network hanging.
    Uses a strict 30-second timeout so it will NEVER load infinitely.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API Key not found.")
    
    # Read the mock database directly
    context_text = ""
    data_dir = "data"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.txt'):
                with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                    context_text += f.read() + "\n\n"
                    
    prompt_text = f"""You are Kisan Saathi, an expert agricultural assistant for Indian farmers. 
Your goal is to provide accurate, grounded advice based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
1. ONLY answer questions related to agriculture, farming, crops, soil, irrigation, pests, and government farmer schemes.
2. If the user asks a question COMPLETELY UNRELATED to agriculture, you MUST reply: "I am an agricultural assistant. I can only answer questions related to farming and crop advisory."
3. If the answer to an agricultural question is NOT found in the context provided below, you MUST reply: "I'm sorry, but I do not have enough information in my knowledge base to answer that." Do NOT guess or make up information.
4. Keep your answers simple, empathetic, and highly actionable for a farmer.
5. You MUST reply in the following language: {language}.

Context from Database:
{context_text}

Question: {question}
Helpful Answer:"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.1}
    }
    
    # Make a direct HTTP REST call with a timeout
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    
    result_json = response.json()
    try:
        answer = result_json['candidates'][0]['content']['parts'][0]['text']
        return answer
    except (KeyError, IndexError):
        return "I'm sorry, I couldn't generate a response from the API. Please try again."
