from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

pdf = "D:\Projects\Learning\Rag App\documents\Atomic habits ( PDFDrive ).pdf"
loader = WebBaseLoader(url)
docs = loader.load()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that Summarizes a text based on the provided context."),
        ("human", "{context}")
    ]
)

model = ChatMistralAI(model="mistral-small-2603")

prompt = template.format_prompt(context=docs[0].page_content)

response = model.invoke(prompt.to_messages())

print(response.content)