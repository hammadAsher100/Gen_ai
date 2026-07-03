import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
      "k": 4, 
      "fetch_k": 10,
      "lambda_mult": 0.5
      }
)

llm = ChatMistralAI(model="mistral-medium-2604", temperature=0)

prompt = ChatPromptTemplate.from_messages(
  [
   ("system", """You are a helfpul assistant that provides information about the qeueries asked according to the provided context. 
    If the answer is not present in the context, refine it and combine it with the content relevant to the document.
    But, if it is tottaly out of context, then say "I am sorry, I don't have any information regarding this query." and do not make up any answer."""), 
   ("user", 
    """
    Context : {context}
    Question : {question}
    """)
  ]
)

print(
  """
  Rag System created successfully!
  Press 0 to exit the program
  """
)

while True:
  query = input("\nEnter your query: ")
  if query == "0":
    break
  
  else:
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    response = llm.invoke(prompt.format_prompt(context=context, question=query).to_messages())
    print(f"\nAi: {response.content}")