from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/macbook-neo/"
loader = WebBaseLoader(url)
docs = loader.load()
print(docs[0].page_content)
  