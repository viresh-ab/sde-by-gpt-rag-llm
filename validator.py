def validate_schema(real_df, synthetic_df):
    """
    Normalize and align synthetic Q&A data schema to match real data.
    This function is SAFE for LLM-generated text data.
    """

    # Normalize column names
    synthetic_df.columns = [c.strip() for c in synthetic_df.columns]

    real_cols = list(real_df.columns)
    synth_cols = list(synthetic_df.columns)

    # Add missing columns as empty strings (LLM-safe fallback)
    for col in real_cols:
        if col not in synth_cols:
            synthetic_df[col] = ""

    # Drop extra columns & enforce order
    synthetic_df = synthetic_df[real_cols]

    return synthetic_df
