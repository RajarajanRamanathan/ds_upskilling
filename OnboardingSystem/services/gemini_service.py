import os
import json

from google import genai
from prompts.prompts import build_document_qa_prompt

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------

def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable "
            "is not set."
        )

    return genai.Client(
        api_key=api_key
    )

def analyze_document(
    document_text,
    document_type
):

    client = get_gemini_client()

    prompt = build_document_qa_prompt(
        document_text,
        document_type
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    response_text = response.text.strip()

    # Remove Markdown code fences
    if response_text.startswith("```"):

        response_text = (
            response_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    try:

        return json.loads(response_text)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Gemini returned invalid JSON: {e}"
        )