import pandas as pd
import os
import re
from io import StringIO
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BATCH_SIZE = 200  # SAFE size for Q&A text


def generate_qa_synthetic_data(sample_df: pd.DataFrame, rows: int) -> pd.DataFrame:
    """
    Generate large Q&A datasets using safe LLM batching.
    """
    batches = []
    remaining = rows

    while remaining > 0:
        current_batch = min(BATCH_SIZE, remaining)
        df_batch = _generate_single_batch(sample_df, current_batch)
        batches.append(df_batch)
        remaining -= current_batch

    return pd.concat(batches, ignore_index=True)


def _generate_single_batch(sample_df: pd.DataFrame, rows: int) -> pd.DataFrame:
    columns = list(sample_df.columns)
    examples = sample_df.sample(min(5, len(sample_df)), random_state=None)

    prompt = f"""
You are generating synthetic open-ended survey responses.

STRICT RULES:
- Output CSV ONLY
- NO explanations
- NO markdown
- EVERY value must be wrapped in double quotes
- Use commas ONLY as column separators
- Header must be EXACTLY:
{columns}
- Generate EXACTLY {rows} rows (do NOT stop early)
- Each answer must be unique and natural
- Do NOT copy example answers

Style reference ONLY (do not copy):
{examples.to_csv(index=False)}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
    )

    raw = response.choices[0].message.content.strip()

    # Remove any markdown if present
    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()

    try:
        df = pd.read_csv(
            StringIO(raw),
            quotechar='"',
            escapechar='\\',
            engine="python"
        )
    except Exception as e:
        raise ValueError(
            f"Malformed CSV from LLM.\n\nRAW OUTPUT:\n{raw}"
        ) from e

    return df[columns]
