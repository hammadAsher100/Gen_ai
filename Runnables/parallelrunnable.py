from dotenv import load_dotenv
from numpy import short
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

prompt = ChatPromptTemplate.from_template(
  "Expalin the following text in simple words: {text}"
)

model = ChatMistralAI(
  model="mistral-medium-2604", 
  temperature=0
)

output_parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
  "Summarize the following text in one line: {text}"
)
detailed_prompt = ChatPromptTemplate.from_template(
  "Expalin the following text in details: {text}"
)

runnable = RunnableParallel({
  "short": RunnableLambda(lambda x: x['short']) | short_prompt | model | output_parser,
  "detailed": RunnableLambda(lambda x: x['detailed']) | detailed_prompt | model | output_parser
})

result = runnable.invoke({
  "short" : {"text": "Large Language Models"},
  "detailed" : {"text": "Deep Learning and its applications in AI"}
})
print(
  f"""
  Short Answer:
  {result['short']}


  Long Answer:
  {result['detailed']}
  """
)