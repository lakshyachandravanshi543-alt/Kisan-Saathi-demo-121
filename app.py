import streamlit as st
import os

# --- SQLite3 fix for Streamlit Cloud (ChromaDB requirement) ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# --------------------------------------------------------------

from rag_pipeline import load_documents, build_vector_store, get_vector_store, get_rag_chain

st.set_page_config(page_title="Kisan Saathi - Crop Advisory", page_icon="🌾", layout="wide")

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Hide Streamlit default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* App background and fonts */
    .stApp {
        background-color: #f4fcf4;
        font-family: 'Inter', sans-serif;
    }
    
    /* Styled Chat Bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Highlight User vs Assistant */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    
    /* Beautiful expander for sources */
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #2e7d32 !important;
        background-color: #e8f5e9;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Kisan Saathi - RAG Crop Advisory Assistant")
st.markdown("Your smart farming assistant. Ask questions about soil, season, irrigation, fertilizer, pest control, and crop diseases.")

# Sidebar for API Key and Document Upload
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GEMINI_API_KEY"] = api_key

    st.header("🌐 Language Preferences")
    selected_language = st.selectbox("Preferred Response Language", 
                                     ["English", "Hindi", "Marathi", "Punjabi", "Tamil", "Telugu"],
                                     help="The AI will try its best to reply in the selected language.")

    st.header("📂 Knowledge Base")
    uploaded_files = st.file_uploader("Upload more advisory documents (TXT, PDF)", accept_multiple_files=True)
    
    if st.button("Process Documents & Update DB"):
        if not api_key:
            st.error("Please enter the Gemini API Key first.")
        else:
            with st.spinner("Processing documents..."):
                if not os.path.exists("data"):
                    os.makedirs("data")
                
                # Save uploaded files
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        with open(os.path.join("data", uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                
                # Load all docs and build vector store
                docs = load_documents("data")
                if len(docs) > 0:
                    build_vector_store(docs)
                    st.success("Knowledge Base updated successfully!")
                else:
                    st.warning("No documents found to process.")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a farming question (e.g., How to manage Fall Armyworm?)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
            st.error("Please provide a Gemini API Key in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Initialize vector store and RAG chain
                    vectorstore = get_vector_store()
                    if vectorstore is None:
                        # Attempt to build from data dir if missing
                        docs = load_documents("data")
                        if len(docs) > 0:
                            vectorstore = build_vector_store(docs)
                        else:
                            st.error("Knowledge base is empty. Please upload documents or check the 'data' directory.")
                            st.stop()
                    
                    qa_chain = get_rag_chain(vectorstore, language=selected_language)
                    
                    # Get response
                    result = qa_chain.invoke({"query": prompt})
                    answer = result["result"]
                    source_documents = result.get("source_documents", [])
                    
                    st.markdown(answer)
                    
                    # Display Sources
                    if source_documents:
                        with st.expander("📚 Sources & Citing"):
                            for i, doc in enumerate(source_documents):
                                source_name = doc.metadata.get("source", "Unknown")
                                st.markdown(f"**Source {i+1}:** {source_name}")
                                st.caption(doc.page_content[:300] + "...")
                                
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
