# Kisan Saathi - RAG Crop Advisory Assistant 🌾

**Kisan Saathi** is an AI-powered Retrieval-Augmented Generation (RAG) assistant designed to help farmers with crop advisory information. It uses the Google Gemini API to answer queries in simple language (including Hindi) based on a custom knowledge base covering soil, seasons, irrigation, fertilizers, pest control, and crop diseases.

## 🌟 Key Features

- **Custom Knowledge Base**: Ingests agricultural documents (TXT, PDF) to provide domain-specific answers.
- **Source-Aware Answers**: Every response includes a citation from the knowledge base, ensuring accurate, grounded advice.
- **Multilingual Support**: Can understand and respond to queries in Hindi and English.
- **Out-of-Scope Handling**: Politely declines to answer questions unrelated to agriculture and farming.
- **Interactive UI**: A clean, easy-to-use web interface built with Streamlit, including a file uploader to seamlessly expand the knowledge base.

## 🛠️ Technology Stack

- **Frontend UI**: [Streamlit](https://streamlit.io/)
- **LLM & Embeddings**: [Google Gemini API (`gemini-1.5-flash`) & Google Generative AI Embeddings](https://aistudio.google.com/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **RAG Framework**: [LangChain](https://www.langchain.com/)

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your machine. You will also need a Google Gemini API Key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Installation
Clone the repository (or download the files) and navigate to the project directory:

```bash
# It is recommended to use a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install the dependencies
pip install -r requirements.txt
```

### 3. Running the Application
Start the Streamlit app with the following command:

```bash
streamlit run app.py
```

### 4. Usage
1. Open the local URL provided by Streamlit in your browser.
2. Open the sidebar and enter your **Google Gemini API Key**.
3. (Optional) Upload additional PDF or TXT agricultural documents via the sidebar to expand the bot's knowledge.
4. Type your farming-related questions in the chat!

## 🧪 Evaluation & Testing Notes
- **Relevance**: Try asking about specific diseases like "Fall Armyworm" or "Wheat Rusts" based on the default knowledge base.
- **Simple Language**: The assistant is prompted to speak in simple, accessible language suitable for farmers.
- **Citations**: Expand the "📚 Sources & Citing" tab under answers to see exactly which chunk of the document was used.

## 📂 Project Structure
```text
├── app.py               # Main Streamlit frontend application
├── rag_pipeline.py      # Core logic for document chunking, embedding, and retrieval
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables template
└── data/                # Directory containing the knowledge base documents
    └── crop_advisory.txt # Default mock knowledge base
```
