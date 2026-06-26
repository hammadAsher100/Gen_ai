import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIRECTORY = r"D:\Projects\Learning\Vector Store\vectorstore"

