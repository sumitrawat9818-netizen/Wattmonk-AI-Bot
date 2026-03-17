import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. THE KEY CHECK ---
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ Add GOOGLE_API_KEY to Streamlit Secrets!")
    st.stop()

st.set_page_config(page_title="Wattmonk AI Bot", layout="wide")
st.title("🤖 Wattmonk AI Support Assistant")

# --- 2. THE BRAIN (RAG) ---
@st.cache_resource
def load_and_index():
    # Make sure these filenames match GitHub EXACTLY
    files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    docs = []
    for f in files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        st.error(f"❌ PDFs NOT FOUND! I see: {os.listdir('.')}")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    # LOCAL EMBEDDINGS (Safe from API Errors)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

with st.spinner("🔄 Building Knowledge Base..."):
    vectorstore = load_and_index()

# --- 3. THE LLM (STABLE VERSION) ---
# Switching to 'gemini-pro' as it is the most widely supported name
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# --- 4. THE CHAT ---
if query := st.chat_input("Ask about Wattmonk (e.g., What is Zippy?)..."):
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in results])
        
        prompt = f"Answer using the context below.\nContext:\n{context}\n\nQuestion: {query}"
        
        try:
            response = llm.invoke(prompt)
            st.markdown(response.content)
            st.info(f"Source: {results[0].metadata['source']}")
        except Exception as e:
            st.error(f"⚠️ Model Error: {e}")
            st.warning("Try changing line 45 to 'gemini-1.5-pro' if this persists.")