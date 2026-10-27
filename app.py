
"""
Short Description:
DocuMind AI document processing and classification application.

Approach:
1. Accept a PDF or TXT document.
2. Extract its text.
3. Load the trained ML pipeline.
4. Predict the document category.
5. Display extracted text and document statistics.
"""

from pathlib import Path

import joblib
import streamlit as st

from document_processor import extract_text, get_document_stats


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "document_classifier.joblib"

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocuMind AI")
st.subheader("Intelligent Document Understanding")

st.write(
    "Upload a document to extract its text, "
    "analyze its content and predict its category."
)


@st.cache_resource
def load_model():
    """Load the trained ML pipeline once per app session."""
    return joblib.load(MODEL_PATH)


uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"],
)

if not MODEL_PATH.exists():
    st.warning(
        "The classification model is not available yet. "
        "Run: python train_classifier.py"
    )

if uploaded_file is not None:
    try:
        text = extract_text(
            uploaded_file.getvalue(),
            uploaded_file.name,
        )

        stats = get_document_stats(text)

        st.success("Document processed successfully!")

        st.subheader("Document Statistics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Words", stats["words"])
        col2.metric("Characters", stats["characters"])
        col3.metric("Lines", stats["lines"])
        col4.metric("Paragraphs", stats["paragraphs"])

        if MODEL_PATH.exists():
            saved_model = load_model()
            pipeline = saved_model["pipeline"]

            prediction = pipeline.predict([text])[0]

            st.subheader("ML Classification Result")

            st.info(f"Predicted category: **{prediction}**")

            st.caption(
                f"Model: {saved_model['model_name']} | "
                f"Test-set macro F1-score: "
                f"{saved_model['macro_f1']:.3f}"
            )

            st.caption(
                "The predicted category is a model estimate, "
                "not a guarantee of correctness. The test-set "
                "score is not a confidence score for this document."
            )

        st.subheader("Extracted Text")

        st.text_area(
            "Document content",
            value=text,
            height=350,
        )

        st.download_button(
            "Download Extracted Text",
            data=text,
            file_name="extracted_text.txt",
            mime="text/plain",
        )

    except Exception as error:
        st.error(f"Could not process document: {error}")
