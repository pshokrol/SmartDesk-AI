import os
import requests
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("JIRA_BASE_URL")
email = os.getenv("JIRA_EMAIL")
token = os.getenv("JIRA_API_TOKEN")
project_key = os.getenv("JIRA_PROJECT_KEY")

print(f"Base URL: {base_url}")
print(f"Email: {email}")
print(f"Project key: {project_key}")
print(f"Token loaded: {'yes' if token else 'NO - MISSING'}")

# Try a simple authenticated GET request — fetch project details.
# This confirms auth + base URL + project key are all correct together.
response = requests.get(
    f"{base_url}/rest/api/3/project/{project_key}",
    auth=(email, token),
)

print(f"\nStatus code: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Success! Project name: {data['name']}")
else:
    print(f"Error response: {response.text}")