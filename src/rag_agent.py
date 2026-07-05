
from chromadb.base_types import Where
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

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

########### Building a Pydantic model for RAG assessment results #############
from pydantic import BaseModel, Field

class RAGAssessment(BaseModel):
    # only two fields: can_answer and answer:
    can_answer: bool = Field(
    description=(
        "True only if the provided context fully and directly answers the "
        "employee's specific question. False if the context is only "
        "topically related but doesn't contain the actual answer, if it "
        "covers a different case than what was asked (e.g. full-time policy "
        "when the question is about part-time), or if the context is "
        "insufficient in any way. Do not guess or extrapolate beyond what "
        "the context explicitly states."
    )
)
    answer: str = Field(
    description=(
        "The complete answer to the employee's question, written in your own "
        "words based only on the provided context. Leave this as an empty "
        "string if can_answer is False."
    )
)


SYSTEM_PROMPT = """You are SmartDesk AI, the IT & HR help desk assistant for BoldAgent.

You will be given an employee's question and a set of knowledge-base excerpts
retrieved for it.

Rules:
- Answer ONLY using the provided context. Never invent policy details,
  numbers, URLs, or procedures that are not explicitly in the context.
- If the context does not fully and directly answer the specific question
  asked, set can_answer to false — do not guess, extrapolate, or partially
  answer.
- If can_answer is true, write a complete, clear answer in your own words
  (not a copy-paste of the context)."""

def assess_answer(question: str, results: list[tuple[Document, float]]) -> RAGAssessment:
    """Asks the LLM to judge whether the retrieved context answers the question,
    using structured output bound to RAGAssessment. Combines all retrieved
    documents' page_content into one context block, then asks the LLM to
    assess and answer based only on that context.
    
    Example of results input:
    results = [
    (
        Document(
            page_content="Question: How do I reset my password?\nAnswer: Visit the Self-Service Password Portal at passwords.boldagent.com ...",
            metadata={
                "id": "it-001",
                "category": "IT Support",
                "question": "How do I reset my password?",
                "answer": "Visit the Self-Service Password Portal at passwords.boldagent.com. Click 'Forgot Password'..."
            }
        ),
        0.529   # <- distance score, lower = closer match
    ),...
    
    """

    # Build context string from results
    context = "\n\n".join([doc.page_content for doc, _ in results])

    # Create the LLM with structured output
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RAGAssessment)

    
    # Call the LLM with a system message (behavioral rules) + human message (the actual task)
    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", f"Employee question: {question}\n\nRetrieved knowledge-base context:\n{context}"),
    ])
    # Call the LLM with the question and context wo system messsage (= system prompt)):
    # response = llm.invoke(f"Question: {question}\n\nContext:\n{context}")

    return response


def answer_question(vectorstore, question: str) -> RAGAssessment:
    
    """Full RAG pipeline entry point: retrieves KB context, checks retrieval
    confidence (returning early without an LLM call if the match is too weak),
    then asks the LLM to assess and answer if retrieval looked confident."""

    # Step 1: Retrieve with scores
    results = retrieve_with_scores(vectorstore, question)

    # Step 2: Check if confident match
    if not is_confident_match(results):
        return RAGAssessment(can_answer=False, answer="")  # Early return if not confident

    # Step 3: Assess answer with LLM
    assessment = assess_answer(question, results)

    return assessment

