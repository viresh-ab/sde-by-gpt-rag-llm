import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


def scale_with_sdv(seed_df: pd.DataFrame, final_rows: int) -> pd.DataFrame:
    # Build metadata automatically
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(seed_df)

    # Create synthesizer
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True
    )

    # Train on LLM seed data
    synthesizer.fit(seed_df)

    # Generate final synthetic dataset
    synthetic_df = synthesizer.sample(num_rows=final_rows)

    return synthetic_df
