import json
from pathlib import Path

import requests
from PIL.ExifTags import TAGS


DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"


def create_metadata_report(image, filename):
    image_info = {}

    for key, value in image.info.items():
        image_info[str(key)] = str(value)[:4000]

    exif_info = {}

    try:
        exif = image.getexif()

        for key, value in exif.items():
            name = TAGS.get(key, str(key))
            exif_info[str(name)] = str(value)[:2000]

    except Exception:
        pass

    report = {
        "filename": filename,
        "format": image.format,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "image_info": image_info,
        "exif": exif_info,
    }

    report_text = json.dumps(
        report,
        indent=2,
        ensure_ascii=True,
    )


    project_folder = Path(__file__).resolve().parent.parent
    folder = Path("metadata_reports")
    folder.mkdir(exist_ok=True)

    path = folder / "metadata_report.json"

    path.write_text(
        report_text,
        encoding="utf-8",
    )

    return report, report_text


def analyze_metadata_with_deepseek(report_text, api_key):
    system_prompt = """
You are a cautious image-metadata analyst.

You will receive JSON metadata extracted from an image.

Use ONLY the metadata in the JSON.
Do not pretend you saw the image itself.

Your job is to decide whether the metadata contains evidence
that the image was AI-generated.

Important rules:
- Missing metadata is NOT evidence that an image is real.
- Image dimensions alone are NOT AI evidence.
- PNG or JPEG format alone is NOT AI evidence.
- Photoshop or generic editing software alone is NOT proof of AI.
- Only name a generator when the metadata supports it.
- If there is not enough evidence, use "Unknown".
- Strong evidence includes explicit generator names, prompts,
  samplers, step counts, CFG values, workflows, model names,
  or other generation parameters.

Return JSON only in this exact structure:

{
  "usable_for_ai_detection": true,
  "ai_probability": 0,
  "generator": "Unknown",
  "evidence_strength": "none",
  "explanation": "one short explanation",
  "evidence": []
}

Rules for the output:
- usable_for_ai_detection must be true or false.
- ai_probability must be 0 to 100 when usable is true.
- ai_probability must be null when usable is false.
- evidence_strength must be:
  "none", "weak", "medium", or "strong".
- generator can be names such as:
  "Stable Diffusion", "ComfyUI", "AUTOMATIC1111",
  "Midjourney", "DALL-E", "FLUX", "Adobe Firefly",
  "Gemini image tool", "StyleGAN / GAN", or "Unknown".
"""

    response = requests.post(
        DEEPSEEK_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "Analyze this metadata JSON:\n\n"
                        + report_text
                    ),
                },
            ],
            "response_format": {
                "type": "json_object"
            },
            "max_tokens": 700,
        },
        timeout=60,
    )

    response.raise_for_status()

    content = response.json()[
        "choices"
    ][0]["message"]["content"]

    result = json.loads(content)

    usable = bool(
        result.get(
            "usable_for_ai_detection",
            False,
        )
    )

    if not usable:
        return result, None

    probability = result.get(
        "ai_probability"
    )

    if probability is None:
        return result, None

    score = float(probability) / 100
    score = max(0, min(score, 1))

    return result, score


def model_ai_score(label, score):
    if label.lower() == "fake":
        return score

    return 1 - score


def combine_scores(
    model_score,
    metadata_score,
    evidence_strength,
):
    if metadata_score > 0.5:
        return metadata_score

    return model_score * 1.2

    )


def final_label(ai_score):
    if ai_score >= 0.80:
        return "Definetely AI"

    if ai_score >= 0.60:
        return "Probably AI"
    if ai_score >= 0.40:
            return "Possibly AI"
    if ai_score >= 0.2:
            return "Probably Human"
    return "No AI"

   

def choose_generator(
    metadata_result,
    ai_score,
):
    if ai_score < 0.65:
        return "Unknown"

    generator = str(
        metadata_result.get(
            "generator",
            "Unknown",
        )
    )

    if generator.strip() == "":
        return "Unknown"

    return generator

