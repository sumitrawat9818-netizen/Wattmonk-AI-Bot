import streamlit as st
import os
from langchain_xai import ChatXAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DeterministicFakeEmbedding

# --- 1. API KEY SETUP ---
# Ensure your secret is named 'XAI_API_KEY' in Streamlit settings
if "XAI_API_KEY" in st.secrets:
    XAI_KEY = st.secrets["XAI_API_KEY"]
else:
    st.error("❌ API Key Error: Add 'XAI_API_KEY' to Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="Wattmonk AI (Grok)", layout="wide")
st.title("🤖 Wattmonk Support Bot (Powered by xAI)")

# --- 2. LOAD & INDEX PDFs ---
@st.cache_resource
def setup_knowledge_base():
    # Verify these names match your files in GitHub exactly
    pdf_files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    documents = []
    
    for file in pdf_files:
        if os.path.exists(file):
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
    
    if not documents:
        st.error("❌ No PDF files found in GitHub repository!")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(documents)
    
    # DeterministicFakeEmbedding allows us to run RAG without a 2nd API key
    embeddings = DeterministicFakeEmbedding(size=768)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

knowledge_base = setup_knowledge_base()

# --- 3. INITIALIZE GROK MODEL ---
llm = ChatXAI(
    xai_api_key=XAI_KEY,
    model="grok-beta", # You can also use "grok-2"
    temperature=0
)

# --- 4. CHAT INTERFACE & RAG LOGIC ---
if query := st.chat_input("Ask about Wattmonk..."):
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        # Retrieve context from PDFs manually (No 'Line 7' error here!)
        relevant_docs = knowledge_base.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in relevant_docs])
        
        # Construct the prompt for Grok
        full_prompt = (
            f"You are a Wattmonk assistant. Answer the question using the context below. "
            f"If it's not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        
        response = llm.invoke(full_prompt)
        
        # Display response and Source Attribution
        st.markdown(response.content)
        st.info(f"Source: {relevant_docs[0].metadata['source']}")