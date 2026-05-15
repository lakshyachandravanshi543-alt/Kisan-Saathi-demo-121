import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_answer(question, language="English"):
    """
    Direct function to get an answer. 
    Bypasses Langchain LCEL (which causes async deadlocks in Streamlit Cloud).
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.1,
        google_api_key=api_key
    )
    
    # Read the mock database directly
    context_text = ""
    data_dir = "data"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.txt'):
                with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                    context_text += f.read() + "\n\n"
                    
    template = f"""You are Kisan Saathi, an expert agricultural assistant for Indian farmers. 
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
    
    response = llm.invoke(template)
    return response.content
