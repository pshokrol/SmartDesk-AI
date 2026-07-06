import os
import requests
from dotenv import load_dotenv

def create_ticket(email: str, summary: str, description: str, category: str, priority: str = "medium") -> dict:
    """
    Create a Jira ticket in the specified project using the provided details.

    Args:
        email (str): The email of the employee submitting the ticket.
        summary (str): A brief summary of the issue.
        description (str): A detailed description of the issue.
        category (str): The category of the issue (e.g., IT Support, HR).
        priority (str): The priority level of the ticket (default is "medium").

    Returns:
        dict: A dictionary containing the ticket key and URL if successful,
              or an error message if failed.

    Updat ethe code and given that you now need two different URLs for two different purposes (API calls vs. a human-clickable browse link), 
    what's a clean way to store both? Consider adding a new environment variable — maybe JIRA_SITE_URL=https://pshokrol.atlassian.net — alongside your existing JIRA_BASE_URL, and using each for its correct purpose.
    """

    load_dotenv()

    base_url = os.getenv("JIRA_BASE_URL")
    site_url = os.getenv("JIRA_SITE_URL")
    jira_email = os.getenv("JIRA_EMAIL")
    token = os.getenv("JIRA_API_TOKEN")
    project_key = os.getenv("JIRA_PROJECT_KEY")

    # Construct the request payload according to Jira's API requirements
    payload = {
        "fields": {
            "project": {
                "key": project_key
            },
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "text": description,
                                "type": "text"
                            }
                        ]
                    }
                ]
            },
            "issuetype": {
                "name": "Task"
            },
            "labels": [category.upper()],
            "priority": {
                "name": priority.capitalize()
            }
        }
    }

    response = requests.post(
        f"{base_url}/rest/api/3/issue",
        json=payload,
        auth=(jira_email, token)
    )

    if response.status_code in [200, 201]:
        data = response.json()
        ticket_key = data.get("key")
        ticket_url = f"{site_url}/browse/{ticket_key}"
        return {"ticket_key": ticket_key, "ticket_url": ticket_url}
    else:
        return {"error": response.text}
