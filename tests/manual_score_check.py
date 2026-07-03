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


#############  Is confident match test  #############
from src.rag_agent import is_confident_match

print("--- testing is_confident_match ---")
test_queries = [
    "how do I reset my password",       # should be confident
    "where do I park my car",           # should NOT be confident
    "what is the coffee machine schedule",  # should NOT be confident
    "how many sick days do I get",      # should be confident
]
for q in test_queries:
    results = retrieve_with_scores(vectorstore, q)
    confident = is_confident_match(results)
    top_score = results[0][1] if results else None
    print(f"  {q!r}: confident={confident}  top_score={top_score}")


    