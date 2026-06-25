from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.7, 
    max_tokens=2048
)

response = model.invoke("Who is Messi? And which team does he play for? What is his latest achievement?")
print(response.content)  