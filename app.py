import streamlit as st
import os
from langchain_openai import ChatOpenAI 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DeterministicFakeEmbedding

# --- 1. THE API KEY SAFETY NET ---
if "XAI_API_KEY" in st.secrets:
    XAI_KEY = st.secrets["XAI_API_KEY"]
else:
    st.error("❌ XAI_API_KEY missing from Streamlit Secrets!")
    st.stop()

st.set_page_config(page_title="Wattmonk AI Assistant", layout="wide")
st.title("🤖 Wattmonk AI Support Bot")

# --- 2. LOAD & PROCESS DATA ---
@st.cache_resource
def setup_rag():
    pdf_files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    docs = []
    for f in pdf_files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        st.error("❌ No PDFs found in GitHub! Check your filenames.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    embeddings = DeterministicFakeEmbedding(size=768)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

vectorstore = setup_rag()

# --- 3. THE "SAFE MODE" INITIALIZATION ---
# Using 'grok-short' or 'grok-2' as they are the most common 2026 entry points
llm = ChatOpenAI(
    model="grok-2", 
    openai_api_key=XAI_KEY,
    openai_api_base="https://api.x.ai/v1",
    temperature=0
)

# --- 4. CHAT INTERFACE ---
if query := st.chat_input("Ask a question about Wattmonk..."):
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in results])
        
        prompt = f"Answer using the context below.\nContext:\n{context}\n\nQuestion: {query}"
        
        try:
            # Attempt to get a response
            response = llm.invoke(prompt)
            st.markdown(response.content)
            st.info(f"Source: {results[0].metadata['source']}")
        except Exception as e:
            st.error(f"⚠️ Model Error: {e}")
            st.warning("If it says 'Model not found', try changing line 44 to 'grok-2-1212' or 'grok-short'.")