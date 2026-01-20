import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


def scale_with_sdv(seed_df: pd.DataFrame, final_rows: int) -> pd.DataFrame:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(seed_df)

    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True
    )

    synthesizer.fit(seed_df)

    synthetic_df = synthesizer.sample(num_rows=final_rows)
    return synthetic_df
