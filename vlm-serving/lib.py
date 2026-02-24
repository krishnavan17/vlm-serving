import os
from pathlib import Path

from ollama import Client


DEFAULT_MODEL = "qwen3-vl:30b"


def run_vlm(path: Path, prompt: str, model: str = DEFAULT_MODEL) -> str:
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    client = Client(host=ollama_host, trust_env=False)

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(path)],
            }
        ],
    )

    return response.message.content


def describe_image(path: Path) -> str:
    return run_vlm(path, "What is in this image? Be concise.")


def _locate_reference_in_image(path: Path, reference: str) -> str:
    return run_vlm(
        path,
        (
            f"Locate the reference '{reference}' in this image and return ONLY valid JSON. "
            "Use this schema exactly: "
            "{\"reference\": string, \"found\": boolean, \"bbox\": {\"x\": number, \"y\": number, \"width\": number, \"height\": number}, \"coordinates\": \"normalized_0_to_1\"}. "
            "If not found, return: "
            "{\"reference\": string, \"found\": false, \"bbox\": null, \"coordinates\": \"normalized_0_to_1\"}."
        ),
    )


def locate_reference_in_image(path: Path, reference: str) -> str:
    return _locate_reference_in_image(path, reference)


def extract_text_from_image(path: Path) -> str:
    return run_vlm(path, "Extract all visible text from this image. Preserve line breaks and reading order.")


def summarize_document_image(path: Path) -> str:
    return run_vlm(path, "Summarize this document image into 5 concise bullet points.")


def answer_question_about_image(path: Path, question: str) -> str:
    return run_vlm(path, f"Answer this question about the image: {question}")


def extract_structured_fields(path: Path, fields: list[str]) -> str:
    requested_fields = ", ".join(fields)
    return run_vlm(
        path,
        (
            "Extract the requested fields from this image and return ONLY valid JSON. "
            f"Fields: {requested_fields}. "
            "If a field is missing, set it to null."
        ),
    )


def caption_image_detailed(path: Path) -> str:
    return run_vlm(path, "Describe this image in detail. Include key objects, text, layout, and context.")
