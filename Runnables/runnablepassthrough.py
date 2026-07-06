from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", "You are a senior software engineer, that will write code for the user based on their request. You will only respond with code, and nothing else. Only write code in python. In a structurized manner"),
      ("human", "{code}")
    ]
)

explain_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", "You are a senior software engineer, that will explain code for the user based on their request. You will only respond with an explanation of the code, and nothing else. Only write in english"),
      ("human", "Explain the following code in simple terms: {code}")
    ]
)

seq = code_prompt | llm | parser 

seq2 = RunnableParallel(
  {
    "code": RunnablePassthrough(),
    "explanation": explain_prompt | llm | parser
  }
)

chain = seq | seq2

code_input = input("Enter your code request: ")
result = chain.invoke({"code": code_input})
print(result['code'])
print(result['explanation'])
