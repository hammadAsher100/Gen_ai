from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.") 
]


embedding = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="chroma_db",
)

print("Vector database created successfully!")


similarity_retriever = vectorstore.as_retriever(
  search_type="similarity", 
  search_kwargs={"k": 3}
)

print("\nRetrieving similar documents for the query: 'What is gradient descent?'")
query = "What is gradient descent?"
similar_docs = similarity_retriever.invoke(query)
for doc in similar_docs:
    print(f"- {doc.page_content}")
    
mmr_retriever = vectorstore.as_retriever(
  search_type="mmr", 
  search_kwargs={"k": 3, "fetch_k": 5}
)

print("\nRetrieving similar documents for the query: 'What is gradient descent?' using MMR")
query = "What is gradient descent?"

mmr_docs = mmr_retriever.invoke(query)
for doc in mmr_docs:
    print(f"- {doc.page_content}")
  