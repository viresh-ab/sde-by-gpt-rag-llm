def validate_schema(real_df, synthetic_df):
    """
    Normalize and align synthetic data schema to match real data.
    This should NEVER crash the app.
    """

    # Strip whitespace from column names
    synthetic_df.columns = [c.strip() for c in synthetic_df.columns]

    real_cols = list(real_df.columns)
    synth_cols = list(synthetic_df.columns)

    # Missing columns
    missing = [c for c in real_cols if c not in synth_cols]
    if missing:
        raise ValueError(f"Missing columns in synthetic data: {missing}")

    # Drop extra columns
    synthetic_df = synthetic_df[real_cols]

    return synthetic_df
