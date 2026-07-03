from src.ingest import load_qa_pairs, build_documents, build_vectorstore

pairs = load_qa_pairs("data/kb/qa_pairs.json")
docs = build_documents(pairs)

# Build twice in a row — simulates accidentally re-running ingestion
vectorstore = build_vectorstore(docs)
print("After 1st build:", vectorstore._collection.count())

vectorstore = build_vectorstore(docs)
print("After 2nd build:", vectorstore._collection.count())