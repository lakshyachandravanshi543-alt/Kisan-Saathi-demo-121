import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def load_documents(data_dir="data"):
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # This will create and persist the DB
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore

def get_vector_store(persist_directory="./chroma_db"):
    if not os.path.exists(persist_directory):
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )
    return vectorstore

def get_rag_chain(vectorstore):
    # Using gemini-1.5-flash for speed and cost-effectiveness, pro is also an option
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    template = """You are Kisan Saathi, a helpful and friendly agricultural assistant for farmers. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer or if the question is completely out of scope (not related to farming, agriculture, crops, government schemes for farmers, etc.), politely decline to answer.
Always answer in simple, easy-to-understand language. If the user asks in Hindi, reply in Hindi.
Cite the source context (or documents) if possible.

Context: {context}

Question: {question}
Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain
