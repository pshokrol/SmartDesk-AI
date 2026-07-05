
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

key = os.getenv("OPENAI_API_KEY")
if key:
    print(f"Key loaded, starts with {key[:10]}...")
else:
    print("Key not found. check .env file exists and vairable name matches exacrtly")