# 🤖 Wattmonk AI Support Bot

An AI-powered assistant built to provide instant, accurate information regarding Wattmonk’s services, proprietary tools, and operational timelines using Retrieval-Augmented Generation (RAG).

## 🚀 Live Demo
https://wattmonk-ai-bot-98gaas6sfaa2c8k6rjjdrq.streamlit.app/

## 🛠️ Features
- **Intelligent Retrieval**: Extracts data specifically from Wattmonk documentation PDFs.
- **Accurate Answers**: Capable of explaining complex tools like **Zippy** and providing precise turnaround times (e.g., 6-hour plansets).
- **Source Attribution**: Every response identifies which document was used as the source.

## 🧰 Tech Stack
- **Language Model**: Google Gemini 1.5 Flash
- **Orchestration**: LangChain
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Frontend**: Streamlit Cloud

## 📂 File Structure
- `app.py`: The core application logic and RAG pipeline.
- `requirements.txt`: Necessary Python libraries (LangChain, Streamlit, FAISS, etc.).
- `Wattmonk Information.pdf`: Knowledge base documents.

## ⚙️ How to Run Locally
1. **Clone the repo**:
   ```bash
   git clone [https://github.com/sumitrawat9818-netizen/wattmonk-ai-bot.git]
