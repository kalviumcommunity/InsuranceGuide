import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("CHAT_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

prompt = """
Explain Retrieval-Augmented Generation (RAG) in about 100 words.
"""

print("=" * 70)
print("EXPERIMENT 1 - LOW TEMPERATURE (0.2)")
print("=" * 70)

response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.2
    )
)

print(response.text)

print("\n")

print("=" * 70)
print("EXPERIMENT 2 - HIGH TEMPERATURE (1.0)")
print("=" * 70)

response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=1.0
    )
)

print(response.text)

print("\n")

print("=" * 70)
print("EXPERIMENT 3 - MAX OUTPUT TOKENS = 40")
print("=" * 70)

response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=80
    )
)

print(response.text)

print("\n")

print("=" * 70)
print("EXPERIMENT 4 - TOP_P = 0.3")
print("=" * 70)

response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.5,
        top_p=0.3
    )
)

print(response.text)
