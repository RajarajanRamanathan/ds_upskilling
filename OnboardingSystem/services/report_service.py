import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_onboarding_report(
    output_path,
    document_name,
    document_type,
    result,
    review_score
):
    """
    Generate a professional PDF onboarding report.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8
    )

    subheading_style = ParagraphStyle(
        "ReportSubHeading",
        parent=styles["Heading3"],
        fontSize=11,
        spaceBefore=8,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=6
    )

    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10
    )

    story = []


    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "AI Document Onboarding Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Document:</b> {document_name}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Document Type:</b> {document_type}",
            body_style
        )
    )

    story.append(
        Spacer(
            1,
            10
        )
    )


    # ========================================================
    # RISK SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "1. Risk Summary",
            heading_style
        )
    )

    overall_risk = result.get(
        "overall_risk",
        "Unknown"
    )

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

        for item in result.get(
            category,
            []
        ):

            if isinstance(item, dict):

                severity = str(
                    item.get(
                        "severity",
                        "Medium"
                    )
                ).strip().title()

                if severity in counts:

                    counts[severity] += 1


    risk_data = [
        [
            "Overall Risk",
            "Review Score",
            "Critical",
            "High",
            "Medium",
            "Low"
        ],
        [
            overall_risk,
            f"{review_score}/100",
            str(counts["Critical"]),
            str(counts["High"]),
            str(counts["Medium"]),
            str(counts["Low"])
        ]
    ]

    risk_table = Table(
        risk_data,
        colWidths=[
            30 * mm,
            30 * mm,
            25 * mm,
            25 * mm,
            25 * mm,
            25 * mm
        ]
    )

    risk_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            )
        ])
    )

    story.append(
        risk_table
    )

    story.append(
        Spacer(
            1,
            10
        )
    )


    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "2. Executive Summary",
            heading_style
        )
    )

    summary = result.get(
        "summary",
        "No summary available."
    )

    story.append(
        Paragraph(
            str(summary),
            body_style
        )
    )


    # ========================================================
    # KEY TERMS
    # ========================================================

    story.append(
        Paragraph(
            "3. Key Terms",
            heading_style
        )
    )

    key_terms = result.get(
        "key_terms",
        []
    )

    if key_terms:

        data = [
            ["Term", "Value"]
        ]

        for item in key_terms:

            if isinstance(item, dict):

                data.append([
                    str(
                        item.get(
                            "term",
                            ""
                        )
                    ),
                    str(
                        item.get(
                            "value",
                            ""
                        )
                    )
                ])

        table = Table(
            data,
            colWidths=[
                45 * mm,
                125 * mm
            ]
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                )
            ])
        )

        story.append(table)

    else:

        story.append(
            Paragraph(
                "No key terms identified.",
                body_style
            )
        )


    # ========================================================
    # OBLIGATIONS
    # ========================================================

    story.append(
        Paragraph(
            "4. Obligations",
            heading_style
        )
    )

    obligations = result.get(
        "obligations",
        []
    )

    if obligations:

        for item in obligations:

            if isinstance(item, dict):

                party = item.get(
                    "party",
                    "Unknown"
                )

                obligation = item.get(
                    "obligation",
                    ""
                )

                story.append(
                    Paragraph(
                        f"<b>{party}:</b> "
                        f"{obligation}",
                        body_style
                    )
                )

    else:

        story.append(
            Paragraph(
                "No obligations identified.",
                body_style
            )
        )


    # ========================================================
    # FINDING SECTIONS
    # ========================================================

    _add_finding_section(
        story,
        "5. Ambiguities",
        result.get(
            "ambiguities",
            []
        ),
        "issue"
    )

    _add_finding_section(
        story,
        "6. Missing Clauses",
        result.get(
            "missing_clauses",
            []
        ),
        "clause"
    )

    _add_finding_section(
        story,
        "7. Compliance Gaps",
        result.get(
            "compliance_gaps",
            []
        ),
        "issue"
    )

    _add_finding_section(
        story,
        "8. Risks",
        result.get(
            "risks",
            []
        ),
        "risk"
    )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    story.append(
        Paragraph(
            "9. Recommendations",
            heading_style
        )
    )

    recommendation_count = 0

    for category in [
        "ambiguities",
        "missing_clauses",
        "compliance_gaps"
    ]:

        for item in result.get(
            category,
            []
        ):

            if not isinstance(item, dict):
                continue

            recommendation = item.get(
                "recommendation",
                ""
            )

            if recommendation:

                recommendation_count += 1

                story.append(
                    Paragraph(
                        f"{recommendation_count}. "
                        f"{recommendation}",
                        body_style
                    )
                )

    if recommendation_count == 0:

        story.append(
            Paragraph(
                "No recommendations identified.",
                body_style
            )
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is generated "
            "using AI and is intended to support document "
            "review and onboarding workflows. It is not "
            "legal advice or a regulatory certification. "
            "Human review should be performed before making "
            "contractual or compliance decisions.",
            small_style
        )
    )


    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story
    )


# ============================================================
# FINDING SECTION HELPER
# ============================================================

def _add_finding_section(
    story,
    title,
    findings,
    primary_field
):

    story.append(
        Paragraph(
            title,
            ParagraphStyle(
                "FindingHeading",
                parent=getSampleStyleSheet()["Heading2"],
                fontSize=14,
                spaceBefore=12,
                spaceAfter=8
            )
        )
    )

    if not findings:

        story.append(
            Paragraph(
                "No findings identified.",
                ParagraphStyle(
                    "FindingBody",
                    parent=getSampleStyleSheet()["BodyText"],
                    fontSize=9,
                    leading=13
                )
            )
        )

        return


    for index, item in enumerate(
        findings,
        start=1
    ):

        if not isinstance(
            item,
            dict
        ):
            continue

        issue = item.get(
            primary_field,
            ""
        )

        severity = item.get(
            "severity",
            "Medium"
        )

        evidence = item.get(
            "evidence",
            item.get(
                "reason",
                ""
            )
        )

        recommendation = item.get(
            "recommendation",
            ""
        )


        story.append(
            Paragraph(
                f"<b>{index}. {issue}</b>",
                ParagraphStyle(
                    "FindingTitle",
                    parent=getSampleStyleSheet()["BodyText"],
                    fontSize=10,
                    leading=13,
                    spaceBefore=6,
                    spaceAfter=4
                )
            )
        )

        story.append(
            Paragraph(
                f"<b>Severity:</b> {severity}",
                ParagraphStyle(
                    "FindingSeverity",
                    parent=getSampleStyleSheet()["BodyText"],
                    fontSize=9,
                    leading=12
                )
            )
        )

        if evidence:

            story.append(
                Paragraph(
                    f"<b>Evidence / Reason:</b> "
                    f"{evidence}",
                    ParagraphStyle(
                        "FindingEvidence",
                        parent=getSampleStyleSheet()["BodyText"],
                        fontSize=9,
                        leading=12
                    )
                )
            )

        if recommendation:

            story.append(
                Paragraph(
                    f"<b>Recommendation:</b> "
                    f"{recommendation}",
                    ParagraphStyle(
                        "FindingRecommendation",
                        parent=getSampleStyleSheet()["BodyText"],
                        fontSize=9,
                        leading=12,
                        spaceAfter=5
                    )
                )
            )