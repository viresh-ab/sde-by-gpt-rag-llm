import streamlit as st
import pandas as pd

from schema_extractor import extract_schema
from llm_seed_generator import generate_seed_data
from sdv_scaler import scale_with_sdv
from validator import validate_schema

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Synthetic Data Generator | Markelytics Solutions",
    layout="wide"
)

# ---------------- Hide Header ----------------
st.markdown(
    """
    <style>
    .stAppHeader,
    header[data-testid="stHeader"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Markelytics AI | Synthetic Studio")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    real_df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully")
    st.dataframe(real_df.head())

    # ---------------- User Inputs ----------------
    seed_rows = st.number_input(
        "Seed rows (LLM)",
        min_value=100,
        max_value=5000,
        value=500,
        step=100
    )

    final_rows = st.number_input(
        "Final rows",
        min_value=1000,
        max_value=100000,
        value=5000,
        step=500
    )

    epsilon = st.slider(
        "Privacy (ε)",
        min_value=0.1,
        max_value=5.0,
        value=1.0
    )

    # ---------------- Generate Button ----------------
    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating synthetic data..."):
            try:
                # 1️⃣ Extract schema
                schema = extract_schema(real_df)

                # 2️⃣ Generate LLM seed data
                seed_df = generate_seed_data(schema, seed_rows)

                # 3️⃣ Validate schema
                seed_df = validate_schema(real_df, seed_df)

                # 4️⃣ Scale using SDV
                final_df = scale_with_sdv(seed_df, final_rows)

                st.success("Synthetic data generated successfully!")
                st.dataframe(final_df.head())

                # ---------------- Download ----------------
                st.download_button(
                    label="⬇ Download Synthetic CSV",
                    data=final_df.to_csv(index=False),
                    file_name="synthetic_data.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error("Synthetic data generation failed.")
                st.code(str(e))
                st.stop()
else:
    st.info("Please upload a CSV file to continue.")
