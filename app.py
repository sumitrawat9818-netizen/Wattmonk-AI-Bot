import streamlit as st
import os
# We use the most direct imports to avoid "Line 7" version errors
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DeterministicFakeEmbedding

# --- 1. THE API FIX ---
# This looks for your secret and stops the app if it's missing
if "GROQ_API_KEY" in st.secrets:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
else:
    st.error("❌ API Key Error: Go to Streamlit Settings > Secrets and add GROQ_API_KEY")
    st.stop()

st.set_page_config(page_title="Wattmonk AI", layout="wide")
st.title("🤖 Wattmonk Support Assistant")

# --- 2. THE DATA FIX ---
@st.cache_resource
def load_data():
    # List your PDF files exactly as they appear in GitHub
    files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    docs = []
    for f in files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        st.error("❌ No PDF files found! Check filenames in GitHub.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    # Fake embeddings = No second API key needed
    embeddings = DeterministicFakeEmbedding(size=768)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

vectorstore = load_data()

# --- 3. THE LLM FIX (Direct Key Injection) ---
llm = ChatGroq(
    model_name="llama3-8b-8192", 
    groq_api_key=GROQ_KEY, # This fixes the 'API Error'
    temperature=0
)

# --- 4. THE CHAT LOGIC ---
if query := st.chat_input("Ask about Wattmonk..."):
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        # Manual RAG to avoid using the 'Chains' library (Line 7 Error Fix)
        context_docs = vectorstore.similarity_search(query, k=3)
        context_text = "\n\n".join([d.page_content for d in context_docs])
        
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer based ONLY on context:"
        response = llm.invoke(prompt)
        
        st.write(response.content)
        st.info(f"Source: {context_docs[0].metadata['source']}")