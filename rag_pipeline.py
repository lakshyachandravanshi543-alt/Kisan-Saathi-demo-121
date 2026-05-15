import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def load_documents(data_dir="data"):
    """
    RAG Workflow Step 1: Loading
    Ingests agricultural documents (PDFs and Text files) from the specified directory.
    This builds the raw knowledge base before chunking.
    """
    documents = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return documents
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if file.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        elif file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

def build_vector_store(documents, persist_directory="./chroma_db"):
    """
    RAG Workflow Step 2 & 3: Chunking and Embeddings
    Splits the loaded documents into smaller, meaningful chunks and generates
    vector embeddings using Google's Generative AI. It then stores them in a local Chroma vector database.
    """
    # Chunking: Splitting text into manageable 1000-character pieces with 200-character overlap for context preservation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Embeddings: Using local HuggingFace model to avoid API limits and 404 errors
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vector Database: Initializing ChromaDB
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    return vectorstore

def get_vector_store(persist_directory="./chroma_db"):
    if not os.path.exists(persist_directory):
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )
    return vectorstore

def get_rag_chain(vectorstore, language="English"):
    """
    RAG Workflow Step 4 & 5: Retrieval and Generation
    Creates the retrieval QA chain. Retrieves the top-k most relevant chunks from the vector store
    and passes them to the Gemini LLM with a highly constrained prompt to generate an accurate, grounded answer.
    """
    # Using gemini-1.5-flash for speed and cost-effectiveness
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1) # Lower temperature for strictly factual answers
    
    template = f"""You are Kisan Saathi, an expert agricultural assistant for Indian farmers. 
Your goal is to provide accurate, grounded advice based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
1. ONLY answer questions related to agriculture, farming, crops, soil, irrigation, pests, and government farmer schemes.
2. If the user asks a question COMPLETELY UNRELATED to agriculture (e.g., coding, general history, movies, politics), you MUST reply: "I am an agricultural assistant. I can only answer questions related to farming and crop advisory."
3. If the answer to an agricultural question is NOT found in the context provided below, you MUST reply: "I'm sorry, but I do not have enough information in my knowledge base to answer that." Do NOT guess or make up information.
4. Keep your answers simple, empathetic, and highly actionable for a farmer.
5. You MUST reply in the following language: {language}.

Context: {{context}}

Question: {{question}}
Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}), # Retrieve top 4 most relevant chunks
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain
