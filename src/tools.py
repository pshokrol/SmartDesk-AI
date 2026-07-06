from langchain_core.tools import tool
from src.rag_agent import answer_question
from src.jira_tool import create_ticket, get_ticket_status
from src.ingest import load_qa_pairs, build_documents, build_vectorstore

# Build the vectorstore once, when this module is first imported —
# not every time search_knowledge_base is called.
pairs = load_qa_pairs("data/kb/qa_pairs.json")
docs = build_documents(pairs)
vectorstore = build_vectorstore(docs)


@tool
def search_knowledge_base(question: str) -> str:
    """Search the BoldAgent IT & HR knowledge base for an answer to an employee's question.
    Use this whenever an employee asks something that might be answered by company policy
    or IT procedures. Returns the answer if found, or a message saying no confident answer exists."""
    result = answer_question(vectorstore, question)
    if result.can_answer:
        return result.answer
    return "No confident answer found in the knowledge base."


@tool
def create_support_ticket(email: str, summary: str, description: str, category: str, priority: str = "medium") -> dict:
    """Create a Jira ticket in the BoldAgent project for an employee's IT or HR issue.
    Returns a dictionary with the ticket key and URL if successful, or an error message if failed."""
    return create_ticket(email, summary, description, category, priority)


@tool
def check_ticket_status(email: str) -> list[dict]:
    """Retrieve the status of Jira tickets associated with an employee's email.
    Returns a list of dictionaries, each containing ticket key and status."""
    return get_ticket_status(email)