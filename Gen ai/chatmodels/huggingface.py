import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

if "HUGGINGFACEHUB_API_TOKEN" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACEHUB_API_TOKEN"]
elif "HUGGINGFACE_API_KEY" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_API_KEY"]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Who are you?")
print(response.content)