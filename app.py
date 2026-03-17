import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import DeterministicFakeEmbedding

# --- 1. API KEY SETUP ---
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("❌ GROQ_API_KEY missing from Streamlit Secrets!")
    st.stop()

st.title("🤖 Wattmonk AI Assistant")

# --- 2. PREPARE DATA ---
@st.cache_resource
def load_docs():
    files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    docs = []
    for f in files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    embeddings = DeterministicFakeEmbedding(size=768)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()

retriever = load_docs()

# --- 3. THE LLM (GROQ) ---
llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_key)

# --- 4. THE RAG LOGIC (Avoids the 'Line 7' Import) ---
template = """Answer the question based only on the context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. INTERFACE ---
query = st.chat_input("Ask about Wattmonk...")
if query:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        response = rag_chain.invoke(query)
        st.write(response)