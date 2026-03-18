import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. THE KEY ---
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=API_KEY)
else:
    st.error("❌ GOOGLE_API_KEY missing!")
    st.stop()

st.title("🤖 Wattmonk Diagnostic Bot")

# --- 2. THE 404 KILLER: List Available Models ---
st.subheader("Checking your available models...")
try:
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    st.write("✅ Your key has access to:", available_models)
    # Automatically pick the first flash model or fallback
    target_model = available_models[0].split('/')[-1] 
except Exception as e:
    st.error(f"Could not list models: {e}")
    target_model = "gemini-1.5-flash" # Absolute fallback

# --- 3. THE RAG (HuggingFace is safe from 404s) ---
@st.cache_resource
def setup_rag():
    pdf_files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    docs = []
    for f in pdf_files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        st.error("❌ PDFs not found!")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings)

vectorstore = setup_rag()

# --- 4. THE LLM (Using the found model) ---
llm = ChatGoogleGenerativeAI(model=target_model, google_api_key=API_KEY)

if query := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in results])
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        
        try:
            response = llm.invoke(prompt)
            st.markdown(response.content)
            st.info(f"Using model: {target_model}")
        except Exception as e:
            st.error(f"Error: {e}")