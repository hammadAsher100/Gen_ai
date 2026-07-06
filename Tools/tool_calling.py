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
  
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

llm_with_tool = llm.bind_tools([get_text_length])
messages = []
query = HumanMessage("What is the length of the text 'Hello, world!'?")

messages.append(query)
response = llm_with_tool.invoke(messages)
messages.append(response)
print(response)
