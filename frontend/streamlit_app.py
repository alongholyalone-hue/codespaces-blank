import os

import requests
import streamlit as st


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Academic Research Assistant")

st.write(
    "Upload a text-based academic PDF and ask a question. "
    "The application retrieves relevant evidence, extracts "
    "a direct answer, and provides a page-level citation."
)

st.info(
    "The answer is extracted from the uploaded document. "
    "The system does not use outside knowledge."
)

with st.form("document_question_form"):
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Maximum supported file size: 10 MB.",
    )

    question = st.text_input(
        "Question",
        placeholder="For example: What is responsible AI?",
    )

    top_k = st.slider(
        "Search depth",
        min_value=1,
        max_value=10,
        value=3,
        help=(
            "For explanatory questions, the application "
            "automatically examines at least 10 passages "
            "before selecting the final evidence."
        ),
    )

    submitted = st.form_submit_button(
        "Find Answer",
        use_container_width=True,
    )

if submitted:
    if uploaded_file is None:
        st.error("Please upload a PDF.")
        st.stop()

    if not question.strip():
        st.error("Please enter a question.")
        st.stop()

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf",
        )
    }

    form_data = {
        "query": question.strip(),
        "top_k": str(top_k),
    }

    try:
        with st.spinner(
            "Reading the document and finding evidence..."
        ):
            response = requests.post(
                f"{API_BASE_URL}/documents/answer",
                files=files,
                data=form_data,
                timeout=300,
            )

    except requests.ConnectionError:
        st.error(
            "The backend API is not running. "
            "Start FastAPI on port 8000 and try again."
        )
        st.stop()

    except requests.Timeout:
        st.error(
            "The request took too long. "
            "The AI models may still be loading."
        )
        st.stop()

    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        st.stop()

    if response.status_code != 200:
        try:
            error_message = response.json().get(
                "detail",
                "Unknown API error",
            )
        except ValueError:
            error_message = response.text

        st.error(
            f"API error ({response.status_code}): "
            f"{error_message}"
        )
        st.stop()

    result = response.json()

    st.divider()
    st.subheader("Answer")

    if result["answered"]:
        st.success(result["answer"])

        score_column, retrieval_column = st.columns(2)

        answer_confidence = result.get("answer_confidence")

        if answer_confidence is None:
            score_column.metric(
                "Answer mode",
                "Evidence summary",
            )
        else:
            score_column.metric(
                "Answer confidence",
                f"{answer_confidence:.3f}",
            )

        retrieval_column.metric(
            "Retrieval score",
            f"{result['retrieval_score']:.3f}",
        )

        citation = result.get("citation")

        if citation:
            st.subheader("Citation")

            st.write(
                f"**Source:** {citation['source']}"
            )
            st.write(
                f"**Page:** {citation['page_number']}"
            )

            with st.expander(
                "View supporting passage",
                expanded=True,
            ):
                st.write(citation["text"])

    else:
        st.warning(result["answer"])

    with st.expander("Document processing details"):
        st.write(f"**Filename:** {result['source']}")
        st.write(f"**Pages processed:** {result['page_count']}")
        st.write(f"**Text chunks:** {result['chunk_count']}")

st.divider()

st.caption(
    "Limitations: The application supports text-based PDFs. "
    "Scanned documents require OCR, which is not included yet."
)