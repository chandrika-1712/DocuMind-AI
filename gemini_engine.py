
"""
Short Description:
Generate document-grounded answers using the Gemini API.

Approach:
1. Load the Gemini API key from the .env file.
2. Receive the user's question and retrieved passages.
3. Ask Gemini to answer using only the provided evidence.
4. Return the generated answer.
"""

import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


def generate_answer(question: str, passages: list[dict]) -> str:
    """Generate an answer grounded in retrieved document passages."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from the .env file.")

    if not passages:
        return "No relevant passages were found in the document."

    client = genai.Client(api_key=api_key)

    context = "\n\n".join(
        f"Passage {index + 1}:\n{item['passage']}"
        for index, item in enumerate(passages)
    )

    prompt = f"""
You are DocuMind AI, a document question-answering assistant.

Answer the user's question using only the document passages below.

Rules:
- Do not invent facts or use outside knowledge.
- If the passages do not contain enough information, say so clearly.
- Give a clear, concise answer.
- Refer to the relevant passage numbers when possible.

DOCUMENT PASSAGES:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )

    if not response.text:
        return "Gemini did not return an answer. Please try again."

    return response.text.strip()

def summarize_document(text: str) -> str:
    """Generate a concise summary of the uploaded document."""

    if not text.strip():
        raise ValueError("The document contains no text to summarize.")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from the .env file.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are DocuMind AI, a document summarization assistant.

Summarize the document using only its contents.

Requirements:
- Identify the main topic and key points.
- Use clear headings and concise bullet points.
- Preserve important technical terms and facts.
- Do not invent information.
- If the document is incomplete or unclear, mention that limitation.

DOCUMENT:
{text}

SUMMARY:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )

    if not response.text:
        return "Gemini did not return a summary. Please try again."

    return response.text.strip()

