from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(
  max_results=3,
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant
    
    Summarize the following news article in a concise manner, highlighting the key points and main takeaways. Provide the summary in a clear and easy-to-understand format.
    Article: {article}
    """
)

chain = prompt | llm | parser

news_result = search_tool.run(
  "Latest news on AI advancements"
)

result = chain.invoke({"article": news_result})

print(result)
