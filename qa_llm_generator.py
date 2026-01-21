import pandas as pd
import os
from openai import OpenAI
from io import StringIO
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_qa_synthetic_data(sample_df: pd.DataFrame, rows: int) -> pd.DataFrame:
    columns = list(sample_df.columns)
    examples = sample_df.sample(min(5, len(sample_df)), random_state=42)

    prompt = f"""
You are generating synthetic survey responses.

STRICT CSV RULES (VERY IMPORTANT):
- Output CSV ONLY
- NO explanations
- NO markdown
- Every value MUST be wrapped in double quotes
- Use commas ONLY as column separators
- Do NOT add extra columns
- Header must be EXACTLY:
{columns}

Generate {rows} rows.

Example format (for structure only):
"Name","Q1","Q2","Q3",...

Example rows (style reference only, do NOT copy):
{examples.to_csv(index=False)}

Now generate the CSV.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85
    )

    raw = response.choices[0].message.content.strip()

    # Remove markdown if present
    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()

    # --- SAFE CSV PARSING ---
    try:
        df = pd.read_csv(
            StringIO(raw),
            quotechar='"',
            escapechar='\\',
            engine="python"
        )
    except Exception as e:
        raise ValueError(
            "LLM generated malformed CSV.\n\nRAW OUTPUT:\n" + raw
        ) from e

    # Enforce schema order
    df = df[columns]

    return df
