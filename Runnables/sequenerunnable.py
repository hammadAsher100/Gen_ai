from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that provides information about the queries. Return short and consise answers for the query. """), 
    ("user", 
    """
    Context : {context}
    Question : {question}
    """)
])

model = ChatMistralAi(
  model="mistral-medium-2604", 
  temperature=0, 
  output_parser = StrOutputParser()
)



chain = prompt | model 


