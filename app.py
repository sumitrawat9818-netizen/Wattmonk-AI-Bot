import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. API KEY SAFETY ---
# Ensure your Secret name in Streamlit is exactly: GOOGLE_API_KEY
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ GOOGLE_API_KEY missing from Streamlit Secrets!")
    st.stop()

st.set_page_config(page_title="Wattmonk AI Bot", layout="wide")
st.title("🤖 Wattmonk AI Support Bot")

# --- 2. THE PDF PROCESSING ---
@st.cache_resource
def setup_rag():
    # Make sure these filenames match your GitHub exactly
    pdf_files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    docs = []
    for f in pdf_files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        st.error(f"❌ PDFs not found! Found these instead: {os.listdir('.')}")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    # HuggingFace = No API Key, No 404 errors during indexing
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

with st.spinner("Indexing PDFs..."):
    vectorstore = setup_rag()

# --- 3. INITIALIZE GEMINI ---
# Use 'gemini-1.5-flash' - it is the most stable 2026 model name
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=API_KEY, 
    temperature=0
)

# --- 4. CHAT INTERFACE ---
if query := st.chat_input("Ask about Wattmonk..."):
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in results])
        
        prompt = f"Use the context to answer. Context:\n{context}\n\nQuestion: {query}"
        
        try:
            response = llm.invoke(prompt)
            st.markdown(response.content)
            st.info(f"Source: {results[0].metadata['source']}")
        except Exception as e:
            st.error(f"⚠️ Model Error: {e}")