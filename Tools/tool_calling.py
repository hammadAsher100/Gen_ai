from pymupdf import message

from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from rich import print

@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the given text.
    
    Args:
        text (str): The input text whose length is to be calculated.
    
    Returns:
        int: The length of the input text.
    """
    return len(text)
  
tools = {
  "get_text_length": get_text_length
}
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)
llm_with_tool = llm.bind_tools([get_text_length])
messages = []
prompt = input("Enter your prompt: ")
query = HumanMessage(prompt)
messages.append(query)
response = llm_with_tool.invoke(messages)
messages.append(response)
if response.tool_calls:
  tool_name = response.tool_calls[0]["name"]
  tool_message = tools[tool_name].invoke(response.tool_calls[0])
  messages.append(tool_message)

result = llm_with_tool.invoke(messages)
print(result)
