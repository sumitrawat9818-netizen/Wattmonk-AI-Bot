import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. API Configuration
# Replace with your actual Gemini API Key
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

# 2. Build the Knowledge Base 
@st.cache_resource 
def prepare_knowledge_base():
    # Using the exact file names you uploaded
    files = ["Wattmonk (1) (1) (1).pdf", "Wattmonk Information (1).pdf"]
    
    all_docs = []
    for f in files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            all_docs.extend(loader.load())
    
    # Chunking as per assignment technical hints (200-500 tokens)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(chunks, embeddings)

# 3. Streamlit UI
st.set_page_config(page_title="Wattmonk AI Assistant", page_icon="⚡")
st.title("⚡ Wattmonk Knowledge Bot")
st.markdown("Ask me anything about Wattmonk's services, team, or history.")

# Initialize the library
try:
    db = prepare_knowledge_base()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # 4. The RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True 
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = qa_chain({"query": prompt})
            answer = response["result"]
            # Extract source file name for attribution
            source = response["source_documents"][0].metadata['source']
            
            full_response = f"{answer}\n\n**Source:** `{source}`"
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

except Exception as e:
    st.error(f"Please ensure the PDFs are in the folder. Error: {e}")