from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

pdf = "D:\Projects\Learning\Rag App\documents\Atomic habits ( PDFDrive ).pdf"
loader = PyPDFLoader(pdf)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
    )
splits = text_splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that Summarizes a text based on the provided context."),
        ("human", "{context}")
    ]
)

model = ChatMistralAI(model="mistral-small-2603")

prompt = template.format_prompt(context=splits[0].page_content)

response = model.invoke(prompt.to_messages())

print(response.content)

