import os

from dotenv import load_dotenv
from google import genai

from prompts.answer import render_prompt

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

# Context used by the reusable template
context = """
Comprehensive car insurance covers damage to your own vehicle as well as
third-party liabilities. Third-party insurance only covers damages caused
to another person's vehicle, property, or injuries.
"""

# -----------------------------
# Prompt 1 - Using Reusable Template
# -----------------------------
vague_system_prompt = render_prompt(
    context=context,
    question=user_question,
)

# -----------------------------
# Prompt 2 - Reusing the Same Template
# -----------------------------
clear_system_prompt = render_prompt(
    context=context,
    question=user_question,
)

print("=" * 60)
print("PROMPT 1 - TEMPLATE")
print("=" * 60)

response1 = client.models.generate_content(
    model=model_name,
    contents=vague_system_prompt,
)

print(response1.text)

print("\n")

print("=" * 60)
print("PROMPT 2 - TEMPLATE REUSED")
print("=" * 60)

response2 = client.models.generate_content(
    model=model_name,
    contents=clear_system_prompt,
)

print(response2.text)