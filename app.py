import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import DeterministicFakeEmbedding

# --- 1. THE SECRET SAUCE (API SETUP) ---
# We check for 'GROQ_API_KEY' in your Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    key = st.secrets["GROQ_API_KEY"]
else:
    st.error("❌ API Key Error: Please add 'GROQ_API_KEY' to your Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="Wattmonk Support Bot")
st.title("🤖 Wattmonk AI Assistant")

# --- 2. THE DATA (PDF PROCESSING) ---
@st.cache_resource
def setup_rag():
    # Make sure these filenames match your GitHub files EXACTLY
    pdf_files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    docs = []
    for f in pdf_files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        st.error("❌ No PDF files found in GitHub!")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    # We use 'Fake' embeddings so you don't need a 2nd API key for vectors
    embeddings = DeterministicFakeEmbedding(size=768)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()

retriever = setup_rag()

# --- 3. THE BRAIN (GROQ) ---
# We pass the key directly into the model to avoid environment errors
llm = ChatGroq(
    model_name="llama3-8b-8192", 
    api_key=key, 
    temperature=0
)

# --- 4. THE LOGIC ---
prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the context:
{context}
Question: {input}
""")

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# --- 5. THE CHAT ---
if query := st.chat_input("Ask about Wattmonk..."):
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        response = rag_chain.invoke({"input": query})
        st.write(response["answer"])
        # Important for the assignment: Show the source!
        st.info(f"Source: {response['context'][0].metadata['source']}")