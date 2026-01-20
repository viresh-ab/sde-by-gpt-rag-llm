def validate_schema(real_df, synthetic_df):
    if list(real_df.columns) != list(synthetic_df.columns):
        raise ValueError("Schema mismatch between real and synthetic data")
