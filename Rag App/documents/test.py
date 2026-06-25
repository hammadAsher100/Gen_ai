import os
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000, 
  chunk_overlap=10
  )

file_path = os.path.join(os.path.dirname(__file__), "Atomic habits ( PDFDrive ).pdf")
data = PyPDFLoader(file_path)

docs = data.load()

chunks = splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")

print(chunks[0].page_content)