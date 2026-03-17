import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. API KEY SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ GOOGLE_API_KEY missing! Add it to Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="Wattmonk AI Support", layout="wide")
st.title("🤖 Wattmonk AI Support Bot")

# --- 2. THE PDF BRAIN ---
@st.cache_resource
def setup_knowledge_base():
    # Make sure these filenames match your GitHub files EXACTLY
    pdf_files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    docs = []
    for f in pdf_files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        st.error("❌ No PDFs found on GitHub! Check your filenames.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    
    # Gemini's own embeddings (Very accurate for the free tier)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

vectorstore = setup_knowledge_base()

# --- 3. INITIALIZE GEMINI ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- 4. THE CHAT LOGIC ---
if query := st.chat_input("Ask about Wattmonk (e.g., What is Zippy?)..."):
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        # Manual Retrieval to ensure zero 'Line 7' errors
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in results])
        
        prompt = (
            f"You are a Wattmonk assistant. Answer using the context below. "
            f"If it's not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        
        response = llm.invoke(prompt)
        st.markdown(response.content)
        # Source attribution (Crucial for the assessment)
        st.info(f"Source: {results[0].metadata['source']}"))