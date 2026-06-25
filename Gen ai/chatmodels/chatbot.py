import os
import sys
import time
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

model = ChatMistralAI(model="ministral-3b-latest", temperature=0.7, max_tokens=2048)

messages = [
    SystemMessage(content="You are a sarcastic assistant."),
]

print("Starting chatbot engine... Please wait a moment.")
time.sleep(4)
print("Chatbot is ready! Type your message or 'exit' to quit.\n")

while True:
    try:
        prompt = input("You: ").strip()
    except EOFError:
        break
    
    if "python" in prompt.lower() or "chatbot.py" in prompt.lower() or "projects" in prompt.lower():
        continue 

    # 2. Regular Exit checks
    if prompt.lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break
        
    if not prompt:
        continue

    messages.append(HumanMessage(content=prompt))
    
    try:
        response = model.invoke(messages)
        messages.append(AIMessage(content=response.content))
        print(f"Chatbot: {response.content}\n")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        break