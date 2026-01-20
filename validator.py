def validate_schema(real_df, synthetic_df):
    real_cols = list(real_df.columns)
    synth_cols = list(synthetic_df.columns)

    # Normalize column names (strip spaces)
    synthetic_df.columns = [c.strip() for c in synthetic_df.columns]

    if real_cols != list(synthetic_df.columns):
        raise ValueError(
            f"Schema mismatch.\n"
            f"Expected: {real_cols}\n"
            f"Got: {list(synthetic_df.columns)}"
        )

    return synthetic_df
