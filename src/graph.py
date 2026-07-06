from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from src.tools import search_knowledge_base, create_support_ticket, check_ticket_status


def build_agent():
    """
    Creates and returns the SmartDesk AI agent, which is a conversational agent
    that can answer questions from the knowledge base, create Jira tickets, and
    check ticket status. The agent uses a React-style reasoning loop to decide
    which tool to use based on the user's input.
    """

    # Instantiate the model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define the system prompt describing the agent's role and tool usage
    system_prompt = """
    You are SmartDesk AI, the IT & HR help desk assistant for BoldAgent.
    You have access to three tools:
    1. search_knowledge_base(question: str) -> str
        Use this tool to search the BoldAgent IT & HR knowledge base for answers to employee questions. Return the answer if found, or a message saying no confident answer exists.
    2. create_support_ticket(email: str, summary: str, description: str, category: str, priority: str = "medium") -> dict

        Use this tool to create a Jira ticket for an employee's IT or HR issue. Always show the employee a summary of the ticket and get explicit confirmation before creating it.
    3. check_ticket_status(email: str) -> list[dict]

        Use this tool to retrieve the status of Jira tickets associated with an employee's email. Returns a list of dictionaries, each containing ticket key and status.
    """
    
    # Instantiate MemorySaver(), pass it as checkpointer=
    checkpointer = MemorySaver()


    # Create the agent with the specified tools and system prompt
    agent = create_agent(
    model=model,
    tools=[search_knowledge_base, create_support_ticket, check_ticket_status],
    system_prompt=system_prompt,
    checkpointer=checkpointer,
)

    return agent


