import os
from datetime import datetime

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("CHAT_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

system_message = "You are a helpful AI assistant."
user_message = "What is Retrieval-Augmented Generation (RAG)? Explain it in 3 sentences."

try:
    response = client.models.generate_content(
        model=model_name,
        contents=[system_message, user_message]
    )

    print("\nModel Response:\n")
    print(response.text)

    os.makedirs("logs", exist_ok=True)

    with open("logs/chat_log.txt", "a", encoding="utf-8") as log:
        log.write("=" * 60 + "\n")
        log.write(f"Timestamp: {datetime.now()}\n\n")
        log.write("SYSTEM MESSAGE:\n")
        log.write(system_message + "\n\n")
        log.write("USER MESSAGE:\n")
        log.write(user_message + "\n\n")
        log.write("MODEL RESPONSE:\n")
        log.write(response.text + "\n\n")

        if hasattr(response, "usage_metadata"):
            log.write(f"TOKEN USAGE:\n{response.usage_metadata}\n")

        log.write("=" * 60 + "\n\n")

except Exception as e:
    error = str(e)

    if "401" in error or "UNAUTHENTICATED" in error:
        print("Authentication Error (401): Check your API key in the .env file.")

    elif "429" in error or "RESOURCE_EXHAUSTED" in error:
        print("Rate Limit Error (429): Too many requests. Please wait and try again later.")

    else:
        print(f"Unexpected Error: {error}")