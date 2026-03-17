import streamlit as st
import os
# MODERN IMPORTS (Avoids the 'Line 7' legacy chain error)
from langchain_xai import ChatXAI
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

st.set_page_config(page_title="Wattmonk AI Assistant (Grok)", layout="wide")
st.title("🤖 Wattmonk AI Support Bot")

# --- 2. DATA PROCESSING (Bypasses Line 7 Error) ---
@st.cache_resource
def setup_rag():
    # Verify these match your GitHub filenames exactly
    pdf_files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    docs = []
    for f in pdf_files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        st.error("❌ No PDFs found in GitHub! Check filenames.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    
    # Fake embeddings = No need for a 2nd API key for vectors
    embeddings = DeterministicFakeEmbedding(size=768)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

vectorstore = setup_rag()

# --- 3. INITIALIZE GROK ---
# Using the stable 'grok-2' name; if that fails, try 'grok-beta'
llm = ChatXAI(
    xai_api_key=XAI_KEY,
    model="grok-2", 
    temperature=0
)

# --- 4. CHAT INTERFACE & MANUAL RETRIEVAL ---
if query := st.chat_input("Ask a question about Wattmonk..."):
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        # We manually find relevant text to avoid using legacy chains
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in results])
        
        # Construct the prompt directly
        prompt = (
            f"You are a Wattmonk assistant. Use the following context to answer.\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        
        response = llm.invoke(prompt)
        
        # Display Result + Source Citation
        st.markdown(response.content)
        st.info(f"Source: {results[0].metadata['source']}")