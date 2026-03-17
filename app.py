import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# Using a lightweight, local embedding to avoid needing a Google/Gemini key
from langchain_community.embeddings import DeterministicFakeEmbedding

# --- 1. API KEY SETUP ---
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("❌ GROQ_API_KEY missing from Streamlit Secrets!")
    st.stop()

st.set_page_config(page_title="Wattmonk AI Assistant", layout="wide")
st.title("🤖 Wattmonk AI Support Bot")

# --- 2. LOAD & PROCESS DOCUMENTS ---
@st.cache_resource
def prepare_docs():
    # Make sure these filenames match exactly what you uploaded to GitHub
    files = ["Wattmonk Information (1).pdf", "Wattmonk (1) (1) (1).pdf"]
    all_docs = []
    
    for file in files:
        if os.path.exists(file):
            loader = PyPDFLoader(file)
            all_docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(all_docs)
    
    # Initialize Vector Store
    embeddings = DeterministicFakeEmbedding(size=768)
    vectorstore = FAISS.from_documents(final_docs, embeddings)
    return vectorstore.as_retriever()

retriever = prepare_docs()

# --- 3. INITIALIZE GROQ LLM ---
llm = ChatGroq(
    model_name="llama3-8b-8192", 
    groq_api_key=groq_key,
    temperature=0
)

# --- 4. CREATE THE BRAIN (The Chain) ---
system_prompt = (
    "You are an assistant for Wattmonk. Use the following pieces of retrieved context "
    "to answer the question. If you don't know the answer, say that you don't know. "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask me about Wattmonk..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]
        
        # Show the Source (Requirement for the Assignment)
        sources = list(set([doc.metadata['source'] for doc in response['context']]))
        final_response = f"{answer}\n\n**Sources:** {', '.join(sources)}"
        
        st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})