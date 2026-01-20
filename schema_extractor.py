import pandas as pd

def extract_schema(df: pd.DataFrame) -> dict:
    schema = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            schema[col] = {
                "type": "numeric",
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean())
            }
        else:
            schema[col] = {
                "type": "categorical",
                "values": df[col].dropna().unique().tolist()[:20]
            }

    return schema
