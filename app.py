import streamlit as st
import pandas as pd
import tempfile

from schema_extractor import extract_schema
from llm_seed_generator import generate_seed_data
from sdv_scaler import scale_with_sdv
from validator import validate_schema

st.set_page_config(page_title="Synthetic Data Generator |Markelytics Solutions", layout="wide")
st.markdown(
    """
    <style>
    /* ================================
       HIDE STREAMLIT DEFAULT UI
       ================================ */

    /* Hide top Streamlit header */
    .stAppHeader,
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Hide Streamlit Cloud viewer badge & profile container */
    ._container_gzau3_1,
    ._viewerBadge_nim44_23,
    ._profileContainer_gzau3_53 {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
._container_gzau3_1{
 display: none !important;
}
    /* Stable fallbacks (future-proof) */
    div[data-testid="stViewerBadge"],
    div[data-testid="stToolbar"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* Optional: remove extra top padding caused by header */
    .block-container {
        padding-top: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Markelytics AI | Synthetic Studio")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    real_df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully")
    st.dataframe(real_df.head())

    seed_rows = st.number_input("Seed rows (LLM)", 100, 5000, 500)
    final_rows = st.number_input("Final rows", 1000, 100000, 5000)
    epsilon = st.slider("Privacy (ε)", 0.1, 5.0, 1.0)

    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating synthetic data..."):
            schema = extract_schema(real_df)

            seed_df = generate_seed_data(schema, seed_rows)
            validate_schema(real_df, seed_df)

            final_df = scale_with_sdv(seed_df, final_rows)

            st.success("Synthetic data generated successfully!")
            st.dataframe(final_df.head())

            st.download_button(
                "⬇ Download Synthetic CSV",
                final_df.to_csv(index=False),
                "synthetic_data.csv",
                "text/csv"
            )
