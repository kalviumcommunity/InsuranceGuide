import os

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("CHAT_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# ----------------------------
# System Prompt (Task 2)
# ----------------------------

system_prompt = """
You are an internal insurance support assistant.

Role:
- Help employees understand insurance-related policies and procedures.

Scope:
- Answer only insurance-related questions.
- Do not make up information.
- Do not answer unrelated questions.

Constraints:
- Keep responses under 120 words.
- Use a professional and friendly tone.
- Use bullet points whenever appropriate.
- If you do not know the answer, respond:
  "I don't have enough information to answer that. Please consult the official insurance documentation."
"""

# ----------------------------
# User Prompt (Task 1)
# ----------------------------

user_prompt = """
What is the difference between comprehensive car insurance and third-party car insurance?
"""

response = client.models.generate_content(
    model=model_name,
    contents=[
        system_prompt,
        user_prompt
    ]
)

print("\n========== RESPONSE ==========\n")
print(response.text)