import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("CHAT_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

user_question = (
    "What is the difference between comprehensive car insurance "
    "and third-party car insurance?"
)

# -----------------------------
# Prompt 1 - Vague
# -----------------------------
vague_system_prompt = """
You are a helpful AI assistant.
"""

# -----------------------------
# Prompt 2 - Clear and Constrained
# -----------------------------
clear_system_prompt = """
You are an internal insurance support assistant.

Role:
- Help employees understand insurance policies.

Scope:
- Answer only insurance-related questions.
- Do not invent information.

Constraints:
- Keep the answer under 120 words.
- Use bullet points.
- Use a professional and friendly tone.
- If unsure, say:
"I don't have enough information to answer that. Please consult the official insurance documentation."
"""

print("=" * 60)
print("PROMPT 1 - VAGUE")
print("=" * 60)

response1 = client.models.generate_content(
    model=model_name,
    contents=[
        vague_system_prompt,
        user_question,
    ],
)

print(response1.text)

print("\n")

print("=" * 60)
print("PROMPT 2 - CLEAR")
print("=" * 60)

response2 = client.models.generate_content(
    model=model_name,
    contents=[
        clear_system_prompt,
        user_question,
    ],
)

print(response2.text)