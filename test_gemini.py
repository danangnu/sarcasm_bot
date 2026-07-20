import os
import traceback

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "").strip()
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

print("API key loaded:", bool(api_key))
print("API key ending:", api_key[-4:] if api_key else "NONE")
print("Model:", model)

try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: Gemini connection successful",
    )

    print("SUCCESS:")
    print(response.text)

except Exception as exc:
    print("ERROR TYPE:", type(exc).__name__)
    print("ERROR MESSAGE:", str(exc))
    traceback.print_exc()
