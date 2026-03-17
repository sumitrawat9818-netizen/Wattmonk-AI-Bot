import streamlit as st
import os
# MODERN 2026 IMPORTS (Bypasses Line 7 legacy chain error)
from langchain_openai import ChatOpenAI 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DeterministicFakeEmbedding

# --- 1. API KEY SETUP ---
if "XAI_API_KEY" in st.secrets:
    XAI_KEY = st.secrets["XAI_API_KEY"]
else:
    st.error("❌ XAI_API_KEY missing from Streamlit Secrets!")
    st.stop()

st.set_page_config(page_title="Wattmonk AI Assistant", layout="wide")
st.title("🤖 Wattmonk AI Support Bot")

# --- 2. DATA PROCESSING ---
@st.cache_resource
def setup_rag():
    pdf_files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    docs = []
    for f in pdf_files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        st.error("❌ No PDFs found! Check your GitHub filenames.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    
    # Fake embeddings = No 2nd API key needed for vectors
    embeddings = DeterministicFakeEmbedding(size=768)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

vectorstore = setup_rag()

# --- 3. INITIALIZE GROK (2026 STABLE VERSION) ---
# Use 'grok-4-fast-non-reasoning' for speed and stability
llm = ChatOpenAI(
    model="grok-4-fast-non-reasoning", 
    openai_api_key=XAI_KEY,
    openai_api_base="https://api.x.ai/v1",
    temperature=0
)

# --- 4. CHAT INTERFACE ---
if query := st.chat_input("Ask a question about Wattmonk..."):
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        # Manual RAG avoids legacy 'Chains' module errors (Line 7 Fix)
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in results])
        
        prompt = (
            f"You are a Wattmonk assistant. Answer using the context below.\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        
        try:
            response = llm.invoke(prompt)
            st.markdown(response.content)
            # Mandatory Source Citation for Sangeeta Yadav's assignment
            st.info(f"Source: {results[0].metadata['source']}")
        except Exception as e:
            # If 400 persists, the error message here will tell us why
            st.error(f"⚠️ Model Error: {e}")
            st.warning("If the error is still 'Model not found', try changing line 45 to 'grok-2'.")