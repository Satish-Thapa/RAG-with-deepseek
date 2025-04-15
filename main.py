import chromadb

client = chromadb.PersistentClient(
    path="./vectorstore") 


collection = client.get_or_create_collection(name="my_programming_collection")


collection.add(
    documents=[
        "Python is a create interpreted language",
        "Working with files in Python, can be done using a context manager",
        "Type-hints is a great way to document your code"
    ],
    metadatas=[{"page": 2, "paragraph": 6},
               {"page": 100, "paragraph": 8},
               {"page": 150, "paragraph": 1}],
    ids=["1xx", "2xx", "3xx"]
)

result = collection.query(
    query_texts=["Python"],
    n_results=2
)
print(result)
