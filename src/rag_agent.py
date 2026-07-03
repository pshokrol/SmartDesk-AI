
from langchain_core.documents import Document
from langchain_chroma import Chroma

def retrieve_with_scores(vectorstore: Chroma, query: str, k: int = 3) -> list[tuple[Document, float]]:
    """Thin wrapper around Chroma's similarity_search_with_score.
    Returns (Document, distance_score) pairs — lower score means closer match.
    """
    
    # Call the vector store's similarity search with score method
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    
    return results_with_scores                                                                          


def is_confident_match(results: list[tuple[Document, float]], threshold: float = 1.2) -> bool:
    """
    Determines if the best match is confident enough based on a similarity score threshold.
    In this function, if results is empty, it returns False (no-retrieval fallback — nothing came back at all).
    Otherwise, it looks at the top result's score (results are already ordered best-to-worst by Chroma). It returns True if it's below threshold, False otherwise.
    Note the threshold value, picked 1.2 by eyeballing your real numbers — it sits between your worst genuine match (1.103) and best gap false-positive-risk (1.377). 
    This will be adjusted once we test it against more queries, but this is a reasonable starting point grounded in your actual data, not a guess.
    """
    
    if not results:
        return False  # No results at all, so not confident

    best_score = results[0][1]  # Get the score of the best match
    return best_score < threshold  # Return True if it's below the threshold, else False









