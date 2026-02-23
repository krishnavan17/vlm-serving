import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile

from lib import (
    answer_question_about_image,
    caption_image_detailed,
    describe_image,
    extract_structured_fields,
    extract_text_from_image,
    locate_reference_in_image,
    summarize_document_image,
)


app = FastAPI(title="VLM Serving API", version="1.0.0")


def resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"file not found: {path}. "
                "If running in Docker, ensure this path is mounted into the vlm-serving container."
            ),
        )
    return path


def run_action(action):
    try:
        result = action()
        return {"result": result}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {error}") from error


async def save_upload_to_temp(image: UploadFile) -> Path:
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="uploaded image is empty")

    suffix = Path(image.filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(content)
        return Path(temp_file.name)


def raise_bad_request(message: str) -> None:
    raise HTTPException(status_code=400, detail=message)


def normalize_text_field(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise_bad_request(f"{field_name} is required")
    return value.strip()


def normalize_fields(value: Any) -> list[str]:
    if isinstance(value, list):
        fields = [str(field).strip() for field in value if str(field).strip()]
    elif isinstance(value, str):
        fields = [field.strip() for field in value.split(",") if field.strip()]
    else:
        fields = []

    if not fields:
        raise_bad_request("at least one valid field is required")

    return fields


async def parse_image_input(request: Request) -> tuple[Path, dict[str, Any], bool]:
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        image = form.get("image")
        if not isinstance(image, UploadFile):
            raise_bad_request("image file is required for multipart requests (field name: image)")

        temp_path = await save_upload_to_temp(image)
        data = {key: value for key, value in form.items() if key != "image"}
        return temp_path, data, True

    try:
        payload = await request.json()
    except json.JSONDecodeError as error:
        raise_bad_request(f"invalid JSON body: {error}")

    if not isinstance(payload, dict):
        raise_bad_request("request body must be a JSON object")

    image_path = payload.get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        raise_bad_request("image_path is required")

    path = resolve_image_path(image_path.strip())
    return path, payload, False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/describe")
async def describe(request: Request) -> dict[str, str]:
    path, _, is_temp = await parse_image_input(request)
    try:
        return run_action(lambda: describe_image(path))
    finally:
        if is_temp:
            path.unlink(missing_ok=True)


@app.post("/locate-reference")
async def locate_reference(request: Request) -> dict[str, str]:
    path, data, is_temp = await parse_image_input(request)
    reference = normalize_text_field(data.get("reference"), "reference")
    try:
        return run_action(lambda: locate_reference_in_image(path, reference))
    finally:
        if is_temp:
            path.unlink(missing_ok=True)


@app.post("/extract-text")
async def extract_text(request: Request) -> dict[str, str]:
    path, _, is_temp = await parse_image_input(request)
    try:
        return run_action(lambda: extract_text_from_image(path))
    finally:
        if is_temp:
            path.unlink(missing_ok=True)


@app.post("/summarize")
async def summarize(request: Request) -> dict[str, str]:
    path, _, is_temp = await parse_image_input(request)
    try:
        return run_action(lambda: summarize_document_image(path))
    finally:
        if is_temp:
            path.unlink(missing_ok=True)


@app.post("/answer-question")
async def answer_question(request: Request) -> dict[str, str]:
    path, data, is_temp = await parse_image_input(request)
    question = normalize_text_field(data.get("question"), "question")
    try:
        return run_action(lambda: answer_question_about_image(path, question))
    finally:
        if is_temp:
            path.unlink(missing_ok=True)


@app.post("/extract-fields")
async def extract_fields(request: Request) -> dict[str, str]:
    path, data, is_temp = await parse_image_input(request)
    fields = normalize_fields(data.get("fields"))
    try:
        return run_action(lambda: extract_structured_fields(path, fields))
    finally:
        if is_temp:
            path.unlink(missing_ok=True)


@app.post("/caption-detailed")
async def caption_detailed(request: Request) -> dict[str, str]:
    path, _, is_temp = await parse_image_input(request)
    try:
        return run_action(lambda: caption_image_detailed(path))
    finally:
        if is_temp:
            path.unlink(missing_ok=True)