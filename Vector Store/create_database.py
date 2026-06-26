import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# PDF path
pdf_path = r"D:\Projects\Learning\Vector Store\documents\Atomic_habits.pdf"

print("Checking PDF...")
print("Exists:", os.path.exists(pdf_path))

print("\nLoading PDF...")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages.")

print("\nSplitting document...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")

print("\nLoading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("Embedding model loaded.")

print("\nCreating Chroma database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db",
)

print("\nVector database created successfully!")
print(f"Stored {len(chunks)} document chunks.")