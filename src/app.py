from src.graph import build_agent
import gradio as gr


# Build the agent once, outside the function, to avoid rebuilding it on every call
agent = build_agent()

def chat_fn(message: str, history: list) -> str:
    """
    This function is called by Gradio every time a user sends a message.
    It invokes the SmartDesk AI agent with the user's message and returns the agent's response.

    Args:
        message (str): The user's input message.
        history (list): The conversation history (not used in this implementation).

    Returns:
        str: The agent's response to the user's message.
    """
    # Invoke the agent with the user's message
    response = agent.invoke(
    {"messages": [{"role": "user", "content": message}]},
    config={"configurable": {"thread_id": "gradio-session"}}
)

    # Extract and return just the text of the last message in the response
    return response["messages"][-1].content

demo = gr.ChatInterface(
    fn=chat_fn,
    title="SmartDesk AI",
    description="IT & HR help desk assistant for BoldAgent employees.",
    examples=[
        ["How do I reset my password?"],
        ["My monitor has been flickering for two days"],
        ["What's the status of my tickets?"],
    ]
)

if __name__ == "__main__":
    demo.launch()