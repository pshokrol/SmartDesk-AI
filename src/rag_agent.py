
from langchain_core.documents import Document
from langchain_chroma import Chroma

def retrieve_with_scores(vectorstore: Chroma, query: str, k: int = 3) -> list[tuple[Document, float]]:
    """Thin wrapper around Chroma's similarity_search_with_score.
    Returns (Document, distance_score) pairs — lower score means closer match.
    """
    
    # Call the vector store's similarity search with score method
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    
    return results_with_scores                                                                          

