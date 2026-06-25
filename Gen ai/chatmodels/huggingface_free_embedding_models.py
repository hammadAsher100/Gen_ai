from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

texts = ["Hello world", "Hi there", "How are you?"]
embedding_vectors = embeddings.embed_documents(texts)
print(embedding_vectors)