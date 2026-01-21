import streamlit as st
import pandas as pd

from schema_extractor import extract_schema
from qa_llm_generator import generate_qa_synthetic_data
from validator import validate_schema

# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(
    page_title="Markelytics AI | Synthetic Q&A Studio",
    layout="wide"
)

# ---------------------------------------------------
# Hide Streamlit UI (header, toolbar, badges)
# ---------------------------------------------------
st.markdown(
    """
    <style>
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stViewerBadge"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    .block-container {
        padding-top: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# App title
# ---------------------------------------------------
st.title("🧠 Markelytics AI | Synthetic Q&A Studio")
st.caption("LLM-only synthetic generation for open-ended survey data")

# ---------------------------------------------------
# File upload
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Q&A Survey CSV",
    type=["csv"]
)

if uploaded_file:
    try:
        real_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("Failed to read CSV file")
        st.code(str(e))
        st.stop()

    st.success("CSV uploaded successfully")
    st.subheader("Sample Input Data")
    st.dataframe(real_df.head())

    # ---------------------------------------------------
    # Controls
    # ---------------------------------------------------
    final_rows = st.number_input(
        "Number of synthetic responses to generate",
        min_value=50,
        max_value=10000,
        value=500,
        step=50
    )

    # ---------------------------------------------------
    # Generate button
    # ---------------------------------------------------
    if st.button("🚀 Generate Synthetic Q&A Data"):
        with st.spinner("Generating synthetic Q&A responses..."):
            try:
                # Extract schema (column names only)
                schema = extract_schema(real_df)

                # LLM-only generation
                synthetic_df = generate_qa_synthetic_data(
                    sample_df=real_df,
                    rows=final_rows
                )

                # Enforce schema & order
                synthetic_df = validate_schema(real_df, synthetic_df)

                st.success("Synthetic Q&A data generated successfully 🎉")

                st.subheader("Synthetic Data Preview")
                st.dataframe(synthetic_df.head())

                # ---------------------------------------------------
                # Download
                # ---------------------------------------------------
                st.download_button(
                    label="⬇ Download Synthetic CSV",
                    data=synthetic_df.to_csv(index=False),
                    file_name="synthetic_qna_data.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error("Failed to generate synthetic Q&A data")
                st.code(str(e))
                st.stop()
