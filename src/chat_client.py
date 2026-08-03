import json
import os
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts.answer import render_prompt

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

context = """
Retrieval-Augmented Generation (RAG) combines document retrieval with a language model.
Relevant documents are retrieved first and used as context before generating an answer.
Source: insurance-guide-rag-notes
"""

USER_MESSAGE = render_prompt(
    context=context,
    question="Explain Retrieval-Augmented Generation (RAG) in one sentence."
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

    print(parsed_response if not parse_error else parse_error)

except Exception as e:
    print(e)