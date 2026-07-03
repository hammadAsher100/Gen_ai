from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
  "Expalin the following text in simple words: {text}"
)

model = ChatMistralAI(
  model="mistral-medium-2604", 
  temperature=0
)
output_parser = StrOutputParser()
chain = prompt | model | output_parser
result = chain.invoke("Tell me about the Mistral AI model.")
print(result)

