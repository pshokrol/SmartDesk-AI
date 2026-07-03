from src.ingest import load_qa_pairs, build_documents, build_vectorstore

pairs = load_qa_pairs("data/kb/qa_pairs.json")
docs = build_documents(pairs)
vectorstore = build_vectorstore(docs)

queries = [
    "how do I reset my password",
    "where do I park my car",
    "what is the coffee machine schedule",
]

for q in queries:
    print(f"--- {q!r} ---")
    results = vectorstore.similarity_search_with_score(q, k=2)
    for doc, score in results:
        print(f"  score={score:.3f}  id={doc.metadata['id']}  q={doc.metadata['question']}")


#############  Retrieve with scores test  #############

from src.rag_agent import retrieve_with_scores

print("--- testing retrieve_with_scores ---")
results = retrieve_with_scores(vectorstore, "how do I reset my password")
for doc, score in results:
    print(f"  score={score:.3f}  id={doc.metadata['id']}")