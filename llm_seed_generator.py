import pandas as pd
import json
import os
from io import StringIO
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_seed_data(schema: dict, rows: int) -> pd.DataFrame:
    prompt = f"""
You are a synthetic data generator.

Schema:
{json.dumps(schema, indent=2)}

Rules:
- Follow schema strictly
- Maintain logical relationships
- Do NOT copy real data
- Generate {rows} rows
- Output CSV only
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    csv_text = response.choices[0].message.content.strip()
    return pd.read_csv(StringIO(csv_text))
