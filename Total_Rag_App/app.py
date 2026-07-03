import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")
st.write("Upload a PDF, build a vector database, and start asking questions.")

# -----------------------------
# Initialize Session State
# -----------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("Upload PDF")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF",
        type="pdf"
    )

    build = st.button(
        "Create Vector Database",
        use_container_width=True
    )


# -----------------------------
# Build Database
# -----------------------------
if build:

    if uploaded_pdf is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    progress = st.progress(0)

    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    progress.progress(10)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    progress.progress(20)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    progress.progress(40)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    progress.progress(55)

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    progress.progress(75)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    progress.progress(100)

    st.session_state.retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    st.success("Vector database created successfully!")

    os.remove(pdf_path)


# -----------------------------
# Chat Model
# -----------------------------
llm = ChatMistralAI(
    model="mistral-medium-2604",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant that answers questions only from the provided context.

Rules:

1. Answer using the context.

2. If the answer is partially available, combine the relevant information.

3. If the question is completely unrelated to the uploaded document, reply:

"I am sorry, I don't have any information regarding this query."

Do not hallucinate.
"""
        ),
        (
            "user",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)


# -----------------------------
# Chat Window
# -----------------------------
st.divider()

st.subheader("Ask Questions")

if st.session_state.retriever is None:

    st.info("Upload a PDF and create the vector database first.")

else:

    for role, message in st.session_state.chat_history:

        with st.chat_message(role):
            st.markdown(message)

    question = st.chat_input("Ask something about your PDF...")

    if question:

        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.chat_history.append(("user", question))

        with st.spinner("Searching document..."):

            docs = st.session_state.retriever.invoke(question)

            context = "\n".join(
                [doc.page_content for doc in docs]
            )

            response = llm.invoke(
                prompt.format_prompt(
                    context=context,
                    question=question
                ).to_messages()
            )

        with st.chat_message("assistant"):
            st.markdown(response.content)

            with st.expander("Retrieved Context"):

                for i, doc in enumerate(docs, start=1):

                    page = doc.metadata.get("page", "Unknown")

                    st.markdown(f"### Chunk {i}")
                    st.write(f"**Page:** {page + 1}")
                    st.write(doc.page_content)

        st.session_state.chat_history.append(
            ("assistant", response.content)
        )