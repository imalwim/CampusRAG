# CampusRAG - Context-Aware Academic Assistant

CampusRAG is an interactive Retrieval-Augmented Generation (RAG) system designed to answer queries based exclusively on uploaded academic documents, lecture slides, and course syllabi. Built with LangChain, ChromaDB, and OpenAI models.

## Key Features
- **Document Chunking & Vectorization:** Splits large PDF documents into semantic chunks using `RecursiveCharacterTextSplitter`.
- **Vector Search:** Uses `ChromaDB` and high-dimensional embeddings for low-latency context retrieval.
- **Hallucination Prevention:** Constrains responses to the retrieved document context.
- **Modern UI:** Clean, responsive interface built with Streamlit.

## Tech Stack
- **Language:** Python 3.10+
- **LLM Orchestration:** LangChain
- **Vector Store:** ChromaDB
- **Models:** OpenAI Embeddings & GPT-4o-mini
- **Frontend:** Streamlit

## Getting Started

### 1. Clone the repository
\`\`\`bash
git clone https://github.com/imalwim/CampusRAG.git
cd CampusRAG
\`\`\`

### 2. Install dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Run the application
\`\`\`bash
streamlit run app.py
\`\`\`
