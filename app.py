import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Local & Free

# --- 1. API KEY SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ GOOGLE_API_KEY missing from Streamlit Secrets!")
    st.stop()

st.set_page_config(page_title="Wattmonk AI Support", layout="wide")
st.title("🤖 Wattmonk AI Support Bot")

# --- 2. DATA PROCESSING ---
@st.cache_resource
def setup_knowledge_base():
    # Verify these filenames match your GitHub repo!
    pdf_files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    docs = []
    
    for f in pdf_files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        st.error("❌ No PDFs found! Check your filenames on GitHub.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    # Bypasses the 'GoogleGenerativeAIError' by using a local model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# This should now run without the 403/400/API error!
vectorstore = setup_knowledge_base()

# --- 3. INITIALIZE GEMINI (For Chatting only) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- 4. CHAT INTERFACE ---
if query := st.chat_input("Ask about Wattmonk (e.g., What is Zippy?)..."):
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in results])
        
        prompt = (
            f"You are a Wattmonk assistant. Answer using the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        
        response = llm.invoke(prompt)
        st.markdown(response.content)
        st.info(f"Source: {results[0].metadata['source']}")