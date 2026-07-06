import os
import requests
from dotenv import load_dotenv
from src.ticket_store import get_ticket_keys_for_email
from src.ticket_store import link_ticket


def create_ticket(email: str, summary: str, description: str, category: str, priority: str = "medium") -> dict:
    """
    Create a Jira ticket in the specified project using the provided details,
    and record the email-to-ticket link locally for later status lookups.

    Args:
        email (str): The email of the employee submitting the ticket.
        summary (str): A brief summary of the issue.
        description (str): A detailed description of the issue.
        category (str): The category of the issue (e.g., IT Support, HR).
        priority (str): The priority level of the ticket (default is "medium").

    Returns:
        dict: A dictionary containing the ticket key and URL if successful,
            or an error message if failed.
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
            "labels": [category.upper().replace(" ", "_")],  # Convert category to uppercase and replace spaces with underscores
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
        link_ticket(email, ticket_key)   # new line
        return {"ticket_key": ticket_key, "ticket_url": ticket_url}
    else:
        return {"error": response.text}



def get_ticket_status(email: str) -> list[dict]:
    """
    Retrieve the current status of all Jira tickets linked to the given employee email.

    Args:
        email (str): The email of the employee whose tickets are to be checked.

    Returns:
        list[dict]: A list of dictionaries, each containing the ticket key, summary, and current status.
                    If no tickets are found, returns an empty list.
    """

    load_dotenv()

    base_url = os.getenv("JIRA_BASE_URL")
    jira_email = os.getenv("JIRA_EMAIL")
    token = os.getenv("JIRA_API_TOKEN")

    # Get the list of ticket keys linked to this email
    ticket_keys = get_ticket_keys_for_email(email)

    if not ticket_keys:
        return []

    ticket_statuses = []
    for ticket_key in ticket_keys:
        response = requests.get(
            f"{base_url}/rest/api/3/issue/{ticket_key}",
            auth=(jira_email, token)
        )

        if response.status_code == 200:
            data = response.json()
            summary = data["fields"]["summary"]
            status = data["fields"]["status"]["name"]
            ticket_statuses.append({
                "ticket_key": ticket_key,
                "summary": summary,
                "status": status
            })
        else:
            ticket_statuses.append({
                "ticket_key": ticket_key,
                "error": response.text
            })

    return ticket_statuses






