from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI


load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chromas_db",
)

retriever = MultiQueryRetriever.from_llm(
    vectorstore=vectorstore,
    llm=ChatMistralAI(model="mistral-medium-2604", temperature=0),
    search_kwargs={"k": 3, "fetch_k": 5}
)

query = "What is gradient descent?"
similar_docs = retriever.invoke(query)
print(f"\nRetrieving similar documents for the query: '{query}' using MultiQueryRetriever")
for doc in similar_docs:
   print(f"- {doc.page_content}")
   
   
    