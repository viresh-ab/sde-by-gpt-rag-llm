import pandas as pd
import json
import os
from io import StringIO
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_seed_data(schema: dict, rows: int) -> pd.DataFrame:
    columns = list(schema.keys())

    prompt = f"""
You are a synthetic data generator.

IMPORTANT:
- The CSV MUST have EXACTLY these columns
- SAME order
- SAME spelling
- NO extra columns

COLUMNS (in order):
{columns}

COLUMN RULES:
{json.dumps(schema, indent=2)}

Generate {rows} rows.

Output rules:
- Output CSV only
- First row must be header EXACTLY as above
- No explanations
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
