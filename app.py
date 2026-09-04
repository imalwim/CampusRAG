import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="CampusRAG - Document Assistant", layout="wide")
st.title("📚 CampusRAG: Academic Document Assistant")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file and api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Save file temporarily for processing
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing and indexing document..."):
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Set up prompt template
        system_prompt = (
            "You are an academic assistant. Answer the question based ONLY on the provided context.\n"
            "If you cannot find the answer in the context, say that you do not know.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    st.success("Document indexed successfully!")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if query := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            response = rag_chain.invoke({"input": query})
            answer = response["answer"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

elif not api_key:
    st.info("Please enter your OpenAI API key to proceed.")
