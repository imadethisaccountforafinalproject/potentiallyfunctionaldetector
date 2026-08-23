import base64
import io
import json
import requests




DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash-vision-exp"



def check_metadata(image):
    parts = []

    for key, value in image.info.items():
        parts.append(f"{key}: {str(value)[:3000]}")

    try:
        for _, value in image.getexif().items():
            parts.append(str(value))
    except Exception:
        pass

    text = " ".join(parts).lower()

    generators = {
        "automatic1111": "AUTOMATIC1111 / Stable Diffusion",
        "stable diffusion": "Stable Diffusion",
        "comfyui": "ComfyUI",
        "midjourney": "Midjourney",
        "dall-e": "DALL-E",
        "adobe firefly": "Adobe Firefly",
        "firefly": "Adobe Firefly",
        "flux": "FLUX",
        "invokeai": "InvokeAI",
        "fooocus": "Fooocus",
        "gemini": "Gemini image tool",
    }

    for word, generator in generators.items():
        if word in text:
            return 0.98, generator, f'Metadata contains "{word}".'

    keys = [str(key).lower() for key in image.info.keys()]

    if "workflow" in keys:
        return (
            0.90,
            "ComfyUI or another workflow tool",
            "A generation workflow was stored in the image metadata.",
        )

    if "parameters" in keys or "prompt" in keys:
        return (
            0.85,
            "Unknown AI tool",
            "Generation-style parameters or a prompt were stored in the metadata.",
        )

    return None, "Unknown", "No clear AI-generation metadata was found."


def check_with_deepseek(image, api_key):
    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=85,
    )

    image_b64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    prompt = """
You are a cautious image-forensics reviewer.
Estimate whether this image LOOKS AI-generated.

Return ONLY valid JSON in this exact format:

{
  "ai_probability": 0,
  "reason": "one short sentence",
  "generator": "Unknown"
}

ai_probability must be from 0 to 100.

generator should be Unknown unless you have a reasonable visual clue.

Possible generator names include Stable Diffusion, Midjourney, DALL-E,
FLUX, GAN-style, or Unknown.

Do not claim certainty.
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
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
        },
        timeout=90,
    )

    response.raise_for_status()

    text = response.json()["choices"][0]["message"]["content"]

    text = (
        text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    result = json.loads(text)

    score = float(
        result.get("ai_probability", 50)
    ) / 100

    score = max(0, min(score, 1))

    reason = str(
        result.get(
            "reason",
            "No explanation returned.",
        )
    )

    generator = str(
        result.get(
            "generator",
            "Unknown",
        )
    )

    return score, reason, generator
