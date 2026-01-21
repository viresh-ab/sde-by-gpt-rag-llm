import pandas as pd
import os
from openai import OpenAI
from io import StringIO
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_qa_synthetic_data(sample_df: pd.DataFrame, rows: int) -> pd.DataFrame:
    columns = list(sample_df.columns)

    # Take a few example rows to show style (not copied)
    examples = sample_df.sample(min(5, len(sample_df)), random_state=42)

    prompt = f"""
You are generating synthetic survey responses.

IMPORTANT RULES:
- Generate {rows} rows
- Keep answers similar in STYLE and LENGTH to examples
- Do NOT copy any example answer
- Each answer must be unique
- Keep natural language (full sentences)
- Output CSV ONLY
- Header must be exactly:
{columns}

Example responses (for style only):
{examples.to_csv(index=False)}

Now generate synthetic responses.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    text = response.choices[0].message.content.strip()

    # Clean markdown if any
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    df = pd.read_csv(StringIO(text))

    # Force schema order
    df = df[columns]

    return df
