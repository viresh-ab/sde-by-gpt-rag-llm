import pandas as pd
import json
import os
import re
from io import StringIO
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _extract_csv(text: str) -> str:
    """
    Extract clean CSV from LLM output.
    Handles markdown fences and extra text.
    """
    text = text.strip()

    # Remove markdown code fences
    text = re.sub(r"```(?:csv)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```", "", text).strip()

    # Keep only lines that look like CSV
    lines = [line for line in text.splitlines() if "," in line]

    if len(lines) < 2:
        raise ValueError("LLM output does not contain valid CSV")

    return "\n".join(lines)


def generate_seed_data(schema: dict, rows: int) -> pd.DataFrame:
    columns = list(schema.keys())

    prompt = f"""
You are a synthetic data generator.

STRICT RULES:
- Output CSV only
- No explanations
- No markdown
- No extra text
- Header MUST be exactly:
{columns}

Generate {rows} rows.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    raw_text = response.choices[0].message.content
    csv_text = _extract_csv(raw_text)

    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception as e:
        raise ValueError(
            f"Failed to parse CSV from LLM output.\n\nRAW OUTPUT:\n{raw_text}"
        ) from e

    return df
