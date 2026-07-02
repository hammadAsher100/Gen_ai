import arxiv

print(arxiv.__version__)

search = arxiv.Search(
    query="gradient descent",
    max_results=1
)

client = arxiv.Client()

for paper in client.results(search):
    print(paper.title)