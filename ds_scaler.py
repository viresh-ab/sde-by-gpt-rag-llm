import tempfile
import pandas as pd

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

def scale_with_datasynthesizer(seed_df, final_rows, epsilon):
    with tempfile.TemporaryDirectory() as tmp:
        seed_path = f"{tmp}/seed.csv"
        desc_path = f"{tmp}/desc.json"
        out_path = f"{tmp}/synthetic.csv"

        seed_df.to_csv(seed_path, index=False)

        describer = DataDescriber(category_threshold=20)
        describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file=seed_path,
            epsilon=epsilon
        )
        describer.save_dataset_description_to_file(desc_path)

        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(
            final_rows, desc_path
        )
        generator.save_synthetic_dataset_to_file(out_path)

        return pd.read_csv(out_path)
