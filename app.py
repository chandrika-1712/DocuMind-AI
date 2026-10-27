
"""
Short Description:
DocuMind AI - Intelligent Document Understanding and RAG Application.

Approach:
1. Accept a PDF or TXT document and extract its text.
2. Predict the document category using a trained ML pipeline.
3. Retrieve relevant passages using Sentence Transformers.
4. Filter passages using a configurable similarity threshold.
5. Generate a grounded answer using the Gemini API.
6. Display the answer, evidence, document statistics, and extracted text.
"""

import hashlib
from pathlib import Path

import joblib
import streamlit as st

from document_processor import extract_text, get_document_stats
from rag_engine import DocumentRetriever
from gemini_engine import generate_answer, summarize_document


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "document_classifier.joblib"

# Starting value only. Tune this using relevant and unrelated questions.
MIN_SIMILARITY = 0.25


# --------------------------------------------------
# Streamlit page setup
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocuMind AI")
st.subheader("Intelligent Document Understanding")

st.write(
    "Upload a PDF or TXT document to classify its content, "
    "find relevant information, and ask questions using AI."
)


# --------------------------------------------------
# Load reusable resources
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained document classification pipeline."""
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_retriever():
    """Load the semantic retrieval model once."""
    return DocumentRetriever()


# --------------------------------------------------
# Document upload and processing
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.warning(
        "The classification model is unavailable. "
        "Run `python train_classifier.py` to create it."
    )

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT document",
    type=["pdf", "txt"],
)

if uploaded_file is not None:
    try:
        # Read the uploaded file and extract its text.
        file_bytes = uploaded_file.getvalue()
        text = extract_text(file_bytes, uploaded_file.name)

        if not text.strip():
            st.error(
                "No readable text was found. "
                "Please upload a document containing extractable text."
            )
            st.stop()

        # Display document statistics.
        stats = get_document_stats(text)

        st.success("Document processed successfully!")

        st.subheader("📊 Document Statistics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Words", stats["words"])
        col2.metric("Characters", stats["characters"])
        col3.metric("Lines", stats["lines"])
        col4.metric("Paragraphs", stats["paragraphs"])

        
        # --------------------------------------------------
        # AI document summarization
        # --------------------------------------------------

        st.subheader("📝 AI Document Summary")

        if st.button("Summarize Document"):
            try:
                with st.spinner(
                    "Gemini is summarizing your document..."
                ):
                    summary = summarize_document(text)

                st.markdown(summary)

            except Exception as error:
                st.error(
                    f"Could not summarize the document: {error}"
                )


        # --------------------------------------------------
        # ML document classification
        # --------------------------------------------------

        if MODEL_PATH.exists():
            try:
                saved_model = load_model()
                pipeline = saved_model["pipeline"]

                prediction = pipeline.predict([text])[0]

                st.subheader("🏷️ ML Classification Result")
                st.info(f"Predicted category: **{prediction}**")

                st.caption(
                    f"Model: {saved_model['model_name']} | "
                    f"Test-set macro F1-score: "
                    f"{saved_model['macro_f1']:.3f}"
                )

                st.caption(
                    "The predicted category is a model estimate. "
                    "The test-set macro F1-score is not a confidence "
                    "score for this particular document."
                )

            except Exception as error:
                st.warning(
                    f"Document classification failed: {error}"
                )

        # --------------------------------------------------
        # RAG question answering
        # --------------------------------------------------

        st.divider()
        st.subheader("🤖 Ask Questions About Your Document")

        question = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: What are the main points of this document?"
            ),
            key="document_question",
        )

        top_k = st.slider(
            "Number of passages to retrieve",
            min_value=1,
            max_value=5,
            value=3,
        )

        generate_clicked = st.button(
            "Generate Answer",
            type="primary",
        )

        if generate_clicked:
            if not question.strip():
                st.warning("Please enter a question first.")

            else:
                try:
                    retriever = load_retriever()

                    # Re-index when the uploaded document changes.
                    document_id = hashlib.sha256(
                        file_bytes
                    ).hexdigest()

                    if (
                        st.session_state.get("document_id")
                        != document_id
                    ):
                        with st.spinner(
                            "Creating semantic document index..."
                        ):
                            retriever.index_document(text)

                        st.session_state["document_id"] = document_id

                    # Retrieve passages related to the question.
                    with st.spinner(
                        "Searching for relevant passages..."
                    ):
                        passages = retriever.search(
                            question,
                            top_k=top_k,
                        )

                    # Reject unrelated or weakly matched passages.
                    relevant_passages = [
                        item
                        for item in passages
                        if item["similarity"] >= MIN_SIMILARITY
                    ]

                    st.subheader("📚 Supporting Passages")

                    if not relevant_passages:
                        st.warning(
                            "I couldn't find sufficiently relevant "
                            "information in this document to answer "
                            "your question. Try rephrasing it or "
                            "checking whether the information exists "
                            "in the uploaded document."
                        )

                    else:
                        # Generate an answer using only the filtered
                        # passages as evidence.
                        with st.spinner(
                            "Gemini is generating your answer..."
                        ):
                            answer = generate_answer(
                                question,
                                relevant_passages,
                            )

                        st.subheader("💡 AI Answer")
                        st.write(answer)

                        # Show the evidence used by Gemini.
                        for index, item in enumerate(
                            relevant_passages,
                            start=1,
                        ):
                            with st.expander(
                                f"Passage {index}"
                            ):
                                st.write(item["passage"])

                                st.caption(
                                    "Semantic similarity: "
                                    f"{item['similarity']:.3f}"
                                )

                        st.caption(
                            "Similarity scores measure semantic "
                            "relatedness; they are not probabilities "
                            "or guarantees of answer correctness."
                        )

                except Exception as error:
                    st.error(
                        f"Could not generate an answer: {error}"
                    )

        # --------------------------------------------------
        # Extracted text and download
        # --------------------------------------------------

        st.divider()

        with st.expander("📄 View Extracted Text"):
            st.text_area(
                "Document content",
                value=text,
                height=300,
            )

        st.download_button(
            "⬇️ Download Extracted Text",
            data=text,
            file_name="extracted_text.txt",
            mime="text/plain",
        )

    except Exception as error:
        st.error(f"Could not process the document: {error}")
