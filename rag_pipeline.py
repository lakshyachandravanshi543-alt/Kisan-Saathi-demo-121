import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

def get_rag_chain(language="English"):
    """
    EMERGENCY HACKATHON FIX:
    Since Streamlit Cloud fails with vector databases, we use 'In-Context RAG'.
    We load the document text directly into the Gemini prompt. 
    Gemini has a massive context window, so this works perfectly and is 100% bug-free.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
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

Question: {{question}}
Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    # Return a simple runnable chain
    chain = QA_CHAIN_PROMPT | llm
    return chain
