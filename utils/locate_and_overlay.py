#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload an image to /locate-reference and overlay returned bounding box."
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("reference", type=str, help="Reference text to locate")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL for API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <input_stem>_bbox<suffix>)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds (default: 120)",
    )
    return parser.parse_args()


def extract_json_candidate(text: str) -> dict:
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model result")

    return json.loads(match.group(0))


def parse_bbox_payload(result_field) -> tuple[bool, dict]:
    if isinstance(result_field, dict):
        payload = result_field
    elif isinstance(result_field, str):
        payload = extract_json_candidate(result_field)
    else:
        raise ValueError("Unsupported result format from API")

    found = bool(payload.get("found", True))
    bbox = payload.get("bbox")

    if not found:
        return False, {}
    if not isinstance(bbox, dict):
        raise ValueError("BBox is missing or invalid in API response")

    return True, bbox


def bbox_to_pixels(bbox: dict, width: int, height: int) -> tuple[int, int, int, int]:
    if all(k in bbox for k in ("x", "y", "width", "height")):
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["width"])
        h = float(bbox["height"])

        normalized = max(abs(x), abs(y), abs(w), abs(h)) <= 1.5
        if normalized:
            x1 = x * width
            y1 = y * height
            x2 = (x + w) * width
            y2 = (y + h) * height
        else:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
    elif all(k in bbox for k in ("x1", "y1", "x2", "y2")):
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])

        normalized = max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5
        if normalized:
            x1 *= width
            y1 *= height
            x2 *= width
            y2 *= height
    else:
        raise ValueError("BBox must contain either x/y/width/height or x1/y1/x2/y2")

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Computed bbox is invalid after conversion")

    return x1, y1, x2, y2


def call_locate_reference(api_url: str, image_path: Path, reference: str, timeout: int) -> dict:
    endpoint = f"{api_url.rstrip('/')}/locate-reference"

    with image_path.open("rb") as image_file:
        response = requests.post(
            endpoint,
            files={"image": (image_path.name, image_file)},
            data={"reference": reference},
            timeout=timeout,
        )

    response.raise_for_status()
    return response.json()


def main() -> int:
    args = parse_args()

    if not args.image.exists() or not args.image.is_file():
        print(f"Error: input image not found: {args.image}", file=sys.stderr)
        return 1

    output_path = args.output or args.image.with_name(f"{args.image.stem}_bbox{args.image.suffix}")

    try:
        api_response = call_locate_reference(args.api_url, args.image, args.reference, args.timeout)
        result_field = api_response.get("result")
        found, bbox = parse_bbox_payload(result_field)
    except requests.HTTPError as error:
        print(f"HTTP error: {error}", file=sys.stderr)
        try:
            print(error.response.text, file=sys.stderr)
        except Exception:
            pass
        return 1
    except Exception as error:
        print(f"Error parsing API response: {error}", file=sys.stderr)
        return 1

    if not found:
        print("Reference not found. No output image generated.")
        return 2

    try:
        image = Image.open(args.image).convert("RGB")
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = bbox_to_pixels(bbox, image.width, image.height)

        line_width = max(2, min(image.width, image.height) // 300)
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=line_width)

        label = f"{args.reference}"
        text_x = x1
        text_y = max(0, y1 - 14)
        draw.text((text_x, text_y), label, fill="red")

        image.save(output_path)
    except Exception as error:
        print(f"Error drawing bbox: {error}", file=sys.stderr)
        return 1

    print(f"Saved output image: {output_path}")
    print(f"BBox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
