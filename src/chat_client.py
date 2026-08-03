import json
import os
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("CHAT_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

SYSTEM_MESSAGE = (
    "Return only valid JSON with exactly these fields: "
    "answer and source. The answer must be concise and grounded."
)
USER_MESSAGE = (
    "Explain Retrieval-Augmented Generation (RAG) in one sentence and "
    "name the source as 'insurance-guide-rag-notes'."
)
REQUIRED_FIELDS = {"answer", "source"}
STRUCTURED_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "answer": types.Schema(type=types.Type.STRING),
        "source": types.Schema(type=types.Type.STRING),
    },
    required=["answer", "source"],
)


def request_structured_answer(question: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=[SYSTEM_MESSAGE, question],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=STRUCTURED_SCHEMA,
        ),
    )
    return response.text


def parse_structured_response(raw_text: str):
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"Malformed JSON: {exc.msg} at position {exc.pos}."

    if not isinstance(payload, dict):
        return None, "Model response must be a JSON object."

    missing_fields = sorted(REQUIRED_FIELDS.difference(payload.keys()))
    if missing_fields:
        return None, f"Missing required field(s): {', '.join(missing_fields)}"

    return payload, None


try:
    raw_response = request_structured_answer(USER_MESSAGE)
    parsed_response, parse_error = parse_structured_response(raw_response)

    print("\nStructured Model Response:\n")
    if parse_error:
        print(f"Recovery status: {parse_error}")
    else:
        print(parsed_response)

    os.makedirs("logs", exist_ok=True)

    with open("logs/chat_log.txt", "a", encoding="utf-8") as log:
        log.write("=" * 60 + "\n")
        log.write(f"Timestamp: {datetime.now()}\n\n")
        log.write("SYSTEM MESSAGE:\n")
        log.write(SYSTEM_MESSAGE + "\n\n")
        log.write("USER MESSAGE:\n")
        log.write(USER_MESSAGE + "\n\n")
        log.write("MODEL RESPONSE:\n")
        log.write(raw_response + "\n\n")
        log.write("PARSED RESULT:\n")
        log.write(json.dumps(parsed_response or {}, ensure_ascii=True) + "\n\n")

        if parse_error:
            log.write(f"ERROR:\n{parse_error}\n\n")

        log.write("=" * 60 + "\n\n")

except Exception as e:
    error = str(e)

    if "401" in error or "UNAUTHENTICATED" in error:
        print("Authentication Error (401): Check your API key in the .env file.")

    elif "429" in error or "RESOURCE_EXHAUSTED" in error:
        print("Rate Limit Error (429): Too many requests. Please wait and try again later.")

    else:
        print(f"Unexpected Error: {error}")