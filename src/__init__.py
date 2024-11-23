import chromadb

client = chromadb.PersistentClient()

collection = client.get_collection("col")

collection.add