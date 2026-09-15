import os
import streamlit as st

from services.document_parser import (
    extract_text,
    extract_pages
)

from services.gemini_service import (
    analyze_document
)

from services.rag_service import (
    build_vector_store,
    answer_question
)

from services.report_service import (
    generate_onboarding_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Onboarding Assistant",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "document_pages" not in st.session_state:
    st.session_state.document_pages = []

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "document_type" not in st.session_state:
    st.session_state.document_type = ""

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_severity_counts(result):
    """
    Count Critical / High / Medium / Low findings
    across all QA categories.
    """

    counts = {
        "Critical": 0,
        "High": 0,
        "Medium": 0,
        "Low": 0
    }

    categories = [
        "ambiguities",
        "missing_clauses",
        "compliance_gaps",
        "risks"
    ]

    for category in categories:

        findings = result.get(
            category,
            []
        )

        if not isinstance(findings, list):
            continue

        for finding in findings:

            if not isinstance(finding, dict):
                continue

            severity = finding.get(
                "severity",
                "Medium"
            )

            severity = str(
                severity
            ).strip().title()

            if severity in counts:
                counts[severity] += 1

    return counts


def get_severity_icon(severity):

    severity = str(
        severity
    ).strip().title()

    if severity == "Critical":
        return "🔴"

    if severity == "High":
        return "🟠"

    if severity == "Medium":
        return "🟡"

    if severity == "Low":
        return "🟢"

    return "⚪"


def build_review_queue(result):
    """
    Combine all findings into one review queue.
    """

    review_items = []

    # --------------------------------------------------------
    # Ambiguities
    # --------------------------------------------------------

    for item in result.get(
        "ambiguities",
        []
    ):

        if isinstance(item, dict):

            review_items.append({
                "Type": "Ambiguity",
                "Issue": item.get(
                    "issue",
                    ""
                ),
                "Severity": item.get(
                    "severity",
                    "Medium"
                ),
                "Evidence": item.get(
                    "evidence",
                    ""
                ),
                "Recommendation": item.get(
                    "recommendation",
                    ""
                )
            })


    # --------------------------------------------------------
    # Missing Clauses
    # --------------------------------------------------------

    for item in result.get(
        "missing_clauses",
        []
    ):

        if isinstance(item, dict):

            review_items.append({
                "Type": "Missing Clause",
                "Issue": item.get(
                    "clause",
                    ""
                ),
                "Severity": item.get(
                    "severity",
                    "Medium"
                ),
                "Evidence": item.get(
                    "reason",
                    ""
                ),
                "Recommendation": item.get(
                    "recommendation",
                    ""
                )
            })


    # --------------------------------------------------------
    # Compliance Gaps
    # --------------------------------------------------------

    for item in result.get(
        "compliance_gaps",
        []
    ):

        if isinstance(item, dict):

            review_items.append({
                "Type": "Compliance Gap",
                "Issue": item.get(
                    "issue",
                    ""
                ),
                "Severity": item.get(
                    "severity",
                    "Medium"
                ),
                "Evidence": item.get(
                    "evidence",
                    ""
                ),
                "Recommendation": item.get(
                    "recommendation",
                    ""
                )
            })


    # --------------------------------------------------------
    # Risks
    # --------------------------------------------------------

    for item in result.get(
        "risks",
        []
    ):

        if isinstance(item, dict):

            review_items.append({
                "Type": "Risk",
                "Issue": item.get(
                    "risk",
                    ""
                ),
                "Severity": item.get(
                    "severity",
                    "Medium"
                ),
                "Evidence": item.get(
                    "reason",
                    ""
                ),
                "Recommendation": ""
            })


    # --------------------------------------------------------
    # Sort by severity
    # --------------------------------------------------------

    severity_order = {
        "Critical": 1,
        "High": 2,
        "Medium": 3,
        "Low": 4
    }

    review_items.sort(
        key=lambda item:
        severity_order.get(
            str(
                item["Severity"]
            ).strip().title(),
            5
        )
    )

    return review_items


def calculate_review_score(counts):
    """
    Calculate an indicative document review score.

    This is NOT a legal/compliance certification.
    It is only a risk prioritization score.
    """

    critical_weight = (
        counts["Critical"] * 4
    )

    high_weight = (
        counts["High"] * 3
    )

    medium_weight = (
        counts["Medium"] * 2
    )

    low_weight = (
        counts["Low"]
    )

    risk_points = (
        critical_weight
        + high_weight
        + medium_weight
        + low_weight
    )

    score = max(
        0,
        100 - (risk_points * 5)
    )

    return score


# ============================================================
# HEADER
# ============================================================

st.title(
    "📄 AI Document Onboarding Assistant"
)

st.write(
    """
    Upload an SOW, Contract, or SLA to automatically
    extract, analyze, review, and search the document.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    document_type = st.selectbox(
        "Document Type",
        [
            "SOW",
            "Contract",
            "SLA"
        ]
    )

    st.divider()

    st.subheader("Pipeline")

    st.write("✅ Document Upload")
    st.write("✅ PDF / DOCX / TXT Parsing")
    st.write("✅ OCR")
    st.write("✅ Gemini Analysis")
    st.write("✅ Document QA")
    st.write("✅ Page-aware RAG")
    st.write("🚀 Risk Dashboard")

    st.divider()

    st.caption(
        "Gemini + FAISS + Streamlit"
    )


# ============================================================
# 1. DOCUMENT UPLOAD
# ============================================================

st.header("1️⃣ Upload Document")

uploaded_file = st.file_uploader(
    "Choose a document",
    type=[
        "pdf",
        "docx",
        "txt"
    ],
    help=(
        "Supported formats: PDF, DOCX and TXT"
    )
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Selected document: "
        f"**{uploaded_file.name}**"
    )

    upload_directory = "uploads"

    os.makedirs(
        upload_directory,
        exist_ok=True
    )

    file_path = os.path.join(
        upload_directory,
        uploaded_file.name
    )

    if st.button(
        "📥 Process Document",
        type="primary"
    ):

        with st.spinner(
            "Processing document..."
        ):

            try:

                # ------------------------------------------------
                # Save uploaded file
                # ------------------------------------------------

                with open(
                    file_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )


                # ------------------------------------------------
                # Get file type
                # ------------------------------------------------

                file_extension = (
                    os.path.splitext(
                        uploaded_file.name
                    )[1]
                    .lower()
                    .replace(
                        ".",
                        ""
                    )
                )


                # ------------------------------------------------
                # Page-aware extraction
                # ------------------------------------------------

                pages = extract_pages(
                    file_path,
                    file_extension
                )


                # ------------------------------------------------
                # Validate extraction
                # ------------------------------------------------

                if not pages:

                    st.error(
                        "No pages could be extracted."
                    )

                    st.stop()


                # ------------------------------------------------
                # Build combined text
                # ------------------------------------------------

                text_parts = []

                for page in pages:

                    page_number = page.get(
                        "page_number",
                        "Unknown"
                    )

                    source = page.get(
                        "source",
                        "text"
                    )

                    page_text = page.get(
                        "text",
                        ""
                    )

                    text_parts.append(
                        f"--- Page {page_number} "
                        f"[{source}] ---\n"
                        f"{page_text}"
                    )


                extracted_text = (
                    "\n\n".join(
                        text_parts
                    )
                )


                if not extracted_text.strip():

                    st.error(
                        "No text could be extracted "
                        "from the document."
                    )

                    st.stop()


                # ------------------------------------------------
                # Save session state
                # ------------------------------------------------

                st.session_state.document_text = (
                    extracted_text
                )

                st.session_state.document_pages = (
                    pages
                )

                st.session_state.document_name = (
                    uploaded_file.name
                )

                st.session_state.document_type = (
                    document_type
                )

                # Reset previous results
                st.session_state.analysis_result = (
                    None
                )

                st.session_state.rag_ready = (
                    False
                )


                st.success(
                    "Document processed successfully!"
                )


            except Exception as e:

                st.error(
                    f"Document processing failed: {e}"
                )


# ============================================================
# 2. EXTRACTED DOCUMENT
# ============================================================

if st.session_state.document_text:

    st.divider()

    st.header(
        "2️⃣ Extracted Document"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Pages",
            len(
                st.session_state.document_pages
            )
        )

    with col2:

        word_count = len(
            st.session_state.document_text.split()
        )

        st.metric(
            "Words",
            word_count
        )

    with col3:

        character_count = len(
            st.session_state.document_text
        )

        st.metric(
            "Characters",
            character_count
        )


    st.write(
        f"**Document:** "
        f"{st.session_state.document_name}"
    )

    st.write(
        f"**Type:** "
        f"{st.session_state.document_type}"
    )


    # --------------------------------------------------------
    # Page information
    # --------------------------------------------------------

    with st.expander(
        "📄 View Page Information"
    ):

        for page in (
            st.session_state.document_pages
        ):

            page_number = page.get(
                "page_number",
                "Unknown"
            )

            source = page.get(
                "source",
                "text"
            )

            page_text = page.get(
                "text",
                ""
            )

            st.markdown(
                f"**Page {page_number}** "
                f"— `{source}` "
                f"— {len(page_text.split())} words"
            )


    # --------------------------------------------------------
    # Extracted text
    # --------------------------------------------------------

    with st.expander(
        "📖 View Extracted Text"
    ):

        st.text_area(
            "Document Content",
            st.session_state.document_text,
            height=400
        )


# ============================================================
# 3. GEMINI DOCUMENT ANALYSIS
# ============================================================

if st.session_state.document_text:

    st.divider()

    st.header(
        "3️⃣ AI Document Analysis"
    )

    st.write(
        """
        Gemini analyzes the document for key terms,
        obligations, ambiguities, missing clauses,
        compliance gaps, and risks.
        """
    )

    if st.button(
        "🤖 Analyze Document",
        type="primary"
    ):

        with st.spinner(
            "Gemini is analyzing the document..."
        ):

            try:

                result = analyze_document(
                    st.session_state.document_text,
                    st.session_state.document_type
                )

                if not isinstance(
                    result,
                    dict
                ):

                    st.error(
                        "Gemini returned an unexpected response."
                    )

                else:

                    st.session_state.analysis_result = (
                        result
                    )

                    st.success(
                        "Document analysis completed!"
                    )

            except Exception as e:

                st.error(
                    f"Gemini analysis failed: {e}"
                )


# ============================================================
# 4. ANALYSIS RESULTS + DASHBOARD
# ============================================================

if st.session_state.analysis_result:

    result = (
        st.session_state.analysis_result
    )

    st.divider()

    st.header(
        "4️⃣ Analysis Results"
    )


    # ========================================================
    # RISK DASHBOARD
    # ========================================================

    st.subheader(
        "📊 Risk Dashboard"
    )


    overall_risk = result.get(
        "overall_risk",
        "Unknown"
    )

    overall_risk = str(
        overall_risk
    ).strip().title()


    # --------------------------------------------------------
    # Overall risk
    # --------------------------------------------------------

    if overall_risk == "Critical":

        st.error(
            "🔴 Overall Risk: CRITICAL"
        )

    elif overall_risk == "High":

        st.warning(
            "🟠 Overall Risk: HIGH"
        )

    elif overall_risk == "Medium":

        st.warning(
            "🟡 Overall Risk: MEDIUM"
        )

    elif overall_risk == "Low":

        st.success(
            "🟢 Overall Risk: LOW"
        )

    else:

        st.info(
            f"Overall Risk: {overall_risk}"
        )


    # --------------------------------------------------------
    # Severity counts
    # --------------------------------------------------------

    counts = get_severity_counts(
        result
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🔴 Critical",
            counts["Critical"]
        )

    with col2:

        st.metric(
            "🟠 High",
            counts["High"]
        )

    with col3:

        st.metric(
            "🟡 Medium",
            counts["Medium"]
        )

    with col4:

        st.metric(
            "🟢 Low",
            counts["Low"]
        )


    # --------------------------------------------------------
    # Finding summary
    # --------------------------------------------------------

    st.subheader(
        "Finding Summary"
    )

    ambiguity_count = len(
        result.get(
            "ambiguities",
            []
        )
    )

    missing_clause_count = len(
        result.get(
            "missing_clauses",
            []
        )
    )

    compliance_count = len(
        result.get(
            "compliance_gaps",
            []
        )
    )

    risk_count = len(
        result.get(
            "risks",
            []
        )
    )


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "⚠️ Ambiguities",
            ambiguity_count
        )

    with col2:

        st.metric(
            "📋 Missing Clauses",
            missing_clause_count
        )

    with col3:

        st.metric(
            "🛡️ Compliance Gaps",
            compliance_count
        )

    with col4:

        st.metric(
            "🚨 Risks",
            risk_count
        )


    # ========================================================
    # REVIEW SCORE
    # ========================================================

    st.subheader(
        "📈 Document Review Score"
    )

    review_score = calculate_review_score(
        counts
    )

    st.progress(
        review_score / 100
    )

    st.write(
        f"**Review Score: {review_score}/100**"
    )

    st.caption(
        """
        This is an indicative risk-prioritization score
        for document review. It is not a legal or
        regulatory compliance certification.
        """
    )

    if review_score >= 80:

        st.success(
            "Low number of identified issues."
        )

    elif review_score >= 60:

        st.warning(
            "Some issues should be reviewed "
            "before onboarding."
        )

    else:

        st.error(
            "Multiple significant issues "
            "require attention."
        )


    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    st.subheader(
        "📝 Executive Summary"
    )

    summary = result.get(
        "summary",
        "No summary available."
    )

    st.info(
        summary
    )


    # ========================================================
    # REVIEW QUEUE
    # ========================================================

    st.subheader(
        "🎯 Review Queue"
    )

    review_items = build_review_queue(
        result
    )

    if review_items:

        for index, item in enumerate(
            review_items,
            start=1
        ):

            severity = str(
                item["Severity"]
            ).strip().title()

            icon = get_severity_icon(
                severity
            )

            title = (
                f"{icon} {index}. "
                f"{item['Type']} — "
                f"{item['Issue']}"
            )

            with st.expander(
                title
            ):

                st.write(
                    f"**Severity:** {severity}"
                )

                if item["Evidence"]:

                    st.write(
                        f"**Evidence / Reason:** "
                        f"{item['Evidence']}"
                    )

                if item["Recommendation"]:

                    st.write(
                        f"**Recommendation:** "
                        f"{item['Recommendation']}"
                    )

    else:

        st.success(
            "No review items identified."
        )


    # ========================================================
    # DETAILED FINDINGS
    # ========================================================

    st.subheader(
        "🔍 Detailed Findings"
    )


    # --------------------------------------------------------
    # Key Terms
    # --------------------------------------------------------

    with st.expander(
        "🔑 Key Terms"
    ):

        key_terms = result.get(
            "key_terms",
            []
        )

        if key_terms:

            for item in key_terms:

                if isinstance(
                    item,
                    dict
                ):

                    term = item.get(
                        "term",
                        ""
                    )

                    value = item.get(
                        "value",
                        ""
                    )

                    st.markdown(
                        f"**{term}:** {value}"
                    )

                else:

                    st.write(item)

        else:

            st.info(
                "No key terms identified."
            )


    # --------------------------------------------------------
    # Obligations
    # --------------------------------------------------------

    with st.expander(
        "📌 Obligations"
    ):

        obligations = result.get(
            "obligations",
            []
        )

        if obligations:

            for item in obligations:

                if isinstance(
                    item,
                    dict
                ):

                    party = item.get(
                        "party",
                        "Unknown"
                    )

                    obligation = item.get(
                        "obligation",
                        ""
                    )

                    st.markdown(
                        f"**{party}:** "
                        f"{obligation}"
                    )

                else:

                    st.write(item)

        else:

            st.info(
                "No obligations identified."
            )


    # --------------------------------------------------------
    # Ambiguities
    # --------------------------------------------------------

    with st.expander(
        f"⚠️ Ambiguities ({ambiguity_count})"
    ):

        ambiguities = result.get(
            "ambiguities",
            []
        )

        if ambiguities:

            for index, item in enumerate(
                ambiguities,
                start=1
            ):

                if isinstance(
                    item,
                    dict
                ):

                    issue = item.get(
                        "issue",
                        ""
                    )

                    severity = item.get(
                        "severity",
                        "Medium"
                    )

                    evidence = item.get(
                        "evidence",
                        ""
                    )

                    recommendation = item.get(
                        "recommendation",
                        ""
                    )

                    st.markdown(
                        f"### {index}. {issue}"
                    )

                    st.write(
                        f"**Severity:** "
                        f"{get_severity_icon(severity)} "
                        f"{severity}"
                    )

                    if evidence:

                        st.write(
                            f"**Evidence:** "
                            f"{evidence}"
                        )

                    if recommendation:

                        st.write(
                            f"**Recommendation:** "
                            f"{recommendation}"
                        )

        else:

            st.success(
                "No ambiguities identified."
            )


    # --------------------------------------------------------
    # Missing Clauses
    # --------------------------------------------------------

    with st.expander(
        f"📋 Missing Clauses ({missing_clause_count})"
    ):

        missing_clauses = result.get(
            "missing_clauses",
            []
        )

        if missing_clauses:

            for index, item in enumerate(
                missing_clauses,
                start=1
            ):

                if isinstance(
                    item,
                    dict
                ):

                    clause = item.get(
                        "clause",
                        ""
                    )

                    severity = item.get(
                        "severity",
                        "Medium"
                    )

                    reason = item.get(
                        "reason",
                        ""
                    )

                    recommendation = item.get(
                        "recommendation",
                        ""
                    )

                    st.markdown(
                        f"### {index}. {clause}"
                    )

                    st.write(
                        f"**Severity:** "
                        f"{get_severity_icon(severity)} "
                        f"{severity}"
                    )

                    if reason:

                        st.write(
                            f"**Reason:** {reason}"
                        )

                    if recommendation:

                        st.write(
                            f"**Recommendation:** "
                            f"{recommendation}"
                        )

        else:

            st.success(
                "No missing clauses identified."
            )


    # --------------------------------------------------------
    # Compliance Gaps
    # --------------------------------------------------------

    with st.expander(
        f"🛡️ Compliance Gaps ({compliance_count})"
    ):

        compliance_gaps = result.get(
            "compliance_gaps",
            []
        )

        if compliance_gaps:

            for index, item in enumerate(
                compliance_gaps,
                start=1
            ):

                if isinstance(
                    item,
                    dict
                ):

                    issue = item.get(
                        "issue",
                        ""
                    )

                    severity = item.get(
                        "severity",
                        "Medium"
                    )

                    evidence = item.get(
                        "evidence",
                        ""
                    )

                    recommendation = item.get(
                        "recommendation",
                        ""
                    )

                    st.markdown(
                        f"### {index}. {issue}"
                    )

                    st.write(
                        f"**Severity:** "
                        f"{get_severity_icon(severity)} "
                        f"{severity}"
                    )

                    if evidence:

                        st.write(
                            f"**Evidence:** "
                            f"{evidence}"
                        )

                    if recommendation:

                        st.write(
                            f"**Recommendation:** "
                            f"{recommendation}"
                        )

        else:

            st.success(
                "No compliance gaps identified."
            )


    # --------------------------------------------------------
    # Risks
    # --------------------------------------------------------

    with st.expander(
        f"🚨 Risks ({risk_count})"
    ):

        risks = result.get(
            "risks",
            []
        )

        if risks:

            for index, item in enumerate(
                risks,
                start=1
            ):

                if isinstance(
                    item,
                    dict
                ):

                    risk = item.get(
                        "risk",
                        ""
                    )

                    severity = item.get(
                        "severity",
                        "Medium"
                    )

                    reason = item.get(
                        "reason",
                        ""
                    )

                    st.markdown(
                        f"### {index}. {risk}"
                    )

                    st.write(
                        f"**Severity:** "
                        f"{get_severity_icon(severity)} "
                        f"{severity}"
                    )

                    if reason:

                        st.write(
                            f"**Reason:** "
                            f"{reason}"
                        )

        else:

            st.success(
                "No significant risks identified."
            )

# ============================================================
# 4.5 GENERATE ONBOARDING REPORT
# ============================================================

if st.session_state.analysis_result:

    st.divider()

    st.header(
        "📑 Generate Onboarding Report"
    )

    st.write(
        """
        Generate a professional PDF containing the
        AI analysis, risk assessment, findings,
        and recommendations.
        """
    )

    if st.button(
        "📄 Generate PDF Report"
    ):

        with st.spinner(
            "Generating onboarding report..."
        ):

            try:

                result = (
                    st.session_state.analysis_result
                )

                counts = get_severity_counts(
                    result
                )

                review_score = (
                    calculate_review_score(
                        counts
                    )
                )

                report_directory = "reports"

                os.makedirs(
                    report_directory,
                    exist_ok=True
                )

                safe_name = os.path.splitext(
                    st.session_state.document_name
                )[0]

                report_file = os.path.join(
                    report_directory,
                    f"{safe_name}_onboarding_report.pdf"
                )

                generate_onboarding_report(
                    output_path=report_file,
                    document_name=(
                        st.session_state.document_name
                    ),
                    document_type=(
                        st.session_state.document_type
                    ),
                    result=result,
                    review_score=review_score
                )

                st.success(
                    "Onboarding report generated successfully!"
                )

                # ---------------------------------------------
                # Download PDF
                # ---------------------------------------------

                with open(
                    report_file,
                    "rb"
                ) as pdf_file:

                    pdf_bytes = pdf_file.read()

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=os.path.basename(
                        report_file
                    ),
                    mime="application/pdf"
                )

            except Exception as e:

                st.error(
                    f"Failed to generate report: {e}"
                )


# ============================================================
# 5. BUILD FAISS KNOWLEDGE BASE
# ============================================================

if st.session_state.document_pages:

    st.divider()

    st.header(
        "5️⃣ Build Knowledge Base"
    )

    st.write(
        """
        Create page-aware embeddings and store them
        in FAISS for semantic document search.
        """
    )

    if st.button(
        "🧠 Build Knowledge Base"
    ):

        with st.spinner(
            "Creating embeddings and building FAISS index..."
        ):

            try:

                chunk_count = build_vector_store(
                    st.session_state.document_pages,
                    st.session_state.document_name
                )

                st.session_state.rag_ready = True

                st.success(
                    "Knowledge base created successfully! "
                    f"{chunk_count} chunks indexed."
                )

            except Exception as e:

                st.error(
                    f"Failed to build knowledge base: {e}"
                )


# ============================================================
# 5. Build Chroma Knowledge Base
# ============================================================

st.divider()

st.header("🧠 Chroma Knowledge Base")

st.write(
    "Index the processed document in ChromaDB "
    "for semantic search and document Q&A."
)

if st.button("🔎 Build Chroma Knowledge Base"):

    if not st.session_state.document_pages:

        st.warning(
            "Please process a document first."
        )

    else:

        with st.spinner(
            "Creating embeddings and indexing in ChromaDB..."
        ):

            try:

                chunk_count = build_vector_store(
                    st.session_state.document_pages,
                    st.session_state.document_name
                )

                st.session_state.rag_ready = True

                st.success(
                    f"ChromaDB knowledge base created successfully. "
                    f"{chunk_count} chunks indexed."
                )

            except Exception as e:

                st.session_state.rag_ready = False

                st.error(
                    f"Failed to build Chroma knowledge base: {e}"
                )

# ============================================================
# 6. RAG QUESTION ANSWERING
# ============================================================

#if st.session_state.rag_ready:

    #st.divider()

    #st.header(
    #    "6️⃣ Ask Questions About the Document"
    #)

    #st.write(
    #    """
    #    Ask questions in natural language. The system
    #    retrieves relevant document chunks and asks Gemini
    #    to answer using only that context.
    #    """
    #)

    #question = st.text_input(
    #    "Enter your question",
    #    placeholder=(
    #        "Example: What is the termination notice period?"
    #    )
    #)


    #if st.button(
    #   "💬 Ask Question"
    #):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the document and generating answer..."
            ):

                try:

                    answer_result = answer_question(
                        question
                    )


                    # ------------------------------------------------
                    # Answer
                    # ------------------------------------------------

                    st.subheader(
                        "💡 Answer"
                    )

                    answer = answer_result.get(
                        "answer",
                        "No answer generated."
                    )

                    st.write(
                        answer
                    )


                    # ------------------------------------------------
                    # Sources
                    # ------------------------------------------------

                    sources = answer_result.get(
                        "sources",
                        []
                    )

                    if sources:

                        st.subheader(
                            "📚 Sources"
                        )

                        for index, source in enumerate(
                            sources,
                            start=1
                        ):

                            document = source.get(
                                "document",
                                "Unknown"
                            )

                            page_number = source.get(
                                "page_number",
                                "Unknown"
                            )

                            chunk_id = source.get(
                                "chunk_id",
                                "Unknown"
                            )

                            evidence = source.get(
                                "evidence",
                                ""
                            )

                            st.markdown(
                                f"### Source {index}"
                            )

                            st.write(
                                f"**Document:** "
                                f"{document}"
                            )

                            st.write(
                                f"**Page:** "
                                f"{page_number}"
                            )

                            st.write(
                                f"**Chunk:** "
                                f"{chunk_id}"
                            )

                            if evidence:

                                st.info(
                                    evidence
                                )

                            st.divider()

                    else:

                        st.info(
                            "No source information was returned."
                        )


                except Exception as e:

                    st.error(
                        f"Question answering failed: {e}"
                    )

# ============================================================
# RAG Chroma DB Section
# ============================================================

st.divider()

st.header("💬 Ask Questions About the Document")

if not st.session_state.rag_ready:

    st.info(
        "Build the Chroma knowledge base first "
        "before asking questions."
    )

else:

    question = st.text_input(
        "Ask a question about the document",
        placeholder=(
            "Example: What are the payment terms?"
        )
    )

    if st.button("🔍 Ask"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching ChromaDB and generating answer..."
            ):

                try:

                    answer = answer_question(
                        question
                    )

                    st.subheader("Answer")

                    st.write(
                        answer.get(
                            "answer",
                            "No answer generated."
                        )
                    )

                    sources = answer.get(
                        "sources",
                        []
                    )

                    if sources:

                        st.subheader(
                            "📚 Sources"
                        )

                        for source in sources:

                            st.markdown(
                                f"""
**Document:** {source.get('document')}

**Page:** {source.get('page_number')}

**Chunk:** {source.get('chunk_id')}

**Evidence:**

> {source.get('evidence')}
"""
                            )

                except Exception as e:

                    st.error(
                        f"Failed to answer question: {e}"
                    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Document Onboarding Assistant | "
    "Streamlit + Gemini + FAISS"
)