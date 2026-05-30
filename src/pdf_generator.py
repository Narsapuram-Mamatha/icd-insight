"""
pdf_generator.py
────────────────
Generate a clean, professional ICD-10 prediction report PDF using ReportLab.

Usage:
    from src.pdf_generator import generate_report
    generate_report(results, output_path="report_12345.pdf")
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (clinical / professional)
# ─────────────────────────────────────────────────────────────────────────────

DARK_BLUE   = colors.HexColor("#1B3A5C")
MED_BLUE    = colors.HexColor("#2E6DA4")
LIGHT_BLUE  = colors.HexColor("#D6E8F7")
ACCENT_TEAL = colors.HexColor("#17A58D")
WARN_AMBER  = colors.HexColor("#E8A020")
LIGHT_GREY  = colors.HexColor("#F4F6F8")
MID_GREY    = colors.HexColor("#8C9BAB")
RED_LIGHT   = colors.HexColor("#FDE8E8")


# ─────────────────────────────────────────────────────────────────────────────
# Style sheet
# ─────────────────────────────────────────────────────────────────────────────

def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "title", fontSize=18, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontSize=11, textColor=MED_BLUE,
            fontName="Helvetica", spaceAfter=2,
        ),
        "section_header": ParagraphStyle(
            "section_header", fontSize=12, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", fontSize=9, textColor=colors.HexColor("#2C3E50"),
            fontName="Helvetica", leading=13, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small", fontSize=8, textColor=MID_GREY,
            fontName="Helvetica", leading=11,
        ),
        "evidence": ParagraphStyle(
            "evidence", fontSize=8.5, textColor=colors.HexColor("#2C3E50"),
            fontName="Helvetica-Oblique", leading=12,
            leftIndent=12, borderPad=4,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer", fontSize=8, textColor=colors.HexColor("#C0392B"),
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
    }
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# Header / Footer
# ─────────────────────────────────────────────────────────────────────────────

def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4

    # Header bar
    canvas.setFillColor(DARK_BLUE)
    canvas.rect(0, h - 22 * mm, w, 22 * mm, fill=True, stroke=False)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(15 * mm, h - 13 * mm, "ICD-Insight  ·  Clinical ICD-10 Prediction Report")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(w - 15 * mm, h - 13 * mm, f"CONFIDENTIAL — FOR CLINICAL USE ONLY")

    # Footer
    canvas.setFillColor(LIGHT_GREY)
    canvas.rect(0, 0, w, 12 * mm, fill=True, stroke=False)
    canvas.setFillColor(MID_GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(15 * mm, 4 * mm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  ICD-Insight M.Tech Project")
    canvas.drawRightString(w - 15 * mm, 4 * mm, f"Page {doc.page}")

    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    patient_name: str,
    patient_id: str,
    admission_date: str,
    discharge_date: str,
    predictions: List[Dict],
    output_path: str = "icd10_report.pdf",
    model_info: str = "BioClinical ModernBERT + QLoRA",
) -> str:
    """
    Generate a structured ICD-10 prediction PDF report.

    Args:
        patient_name:    Patient's name (or "Unknown")
        patient_id:      MRN or ID (or "Unknown")
        admission_date:  Admission date string
        discharge_date:  Discharge date string
        predictions:     List of dicts:
                           {
                             "icd_code":    "J18.9",
                             "description": "Pneumonia, unspecified",
                             "confidence":  0.87,            # 0–1
                             "evidence":    ["snippet 1 …", "snippet 2 …"]
                           }
        output_path:     Output PDF path
        model_info:      Model description for report footer

    Returns:
        Absolute path to generated PDF
    """
    styles = _build_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=30 * mm,
        bottomMargin=20 * mm,
    )

    story = []

    # ── Patient Info Block ──────────────────────────────────────────────────
    story.append(Paragraph("Clinical ICD-10 Prediction Report", styles["title"]))
    story.append(Spacer(1, 3 * mm))

    info_data = [
        ["Patient Name", patient_name or "Unknown",
         "Patient ID / MRN", patient_id or "Unknown"],
        ["Admission Date", admission_date or "—",
         "Discharge Date", discharge_date or "—"],
    ]
    info_table = Table(info_data, colWidths=[38 * mm, 55 * mm, 38 * mm, 50 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), LIGHT_GREY),
        ("BACKGROUND",  (0, 0), (0, -1), LIGHT_BLUE),
        ("BACKGROUND",  (2, 0), (2, -1), LIGHT_BLUE),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 0), (0, -1), DARK_BLUE),
        ("TEXTCOLOR",   (2, 0), (2, -1), DARK_BLUE),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#C5D5E8")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GREY, colors.white]),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 5 * mm))

    story.append(HRFlowable(width="100%", thickness=1.5, color=MED_BLUE))
    story.append(Spacer(1, 4 * mm))

    # ── Summary Table ────────────────────────────────────────────────────────
    story.append(Paragraph("Predicted ICD-10-CM Codes", styles["section_header"]))

    table_data = [["ICD-10 Code", "Description", "Confidence", "Confidence Bar"]]
    for pred in sorted(predictions, key=lambda x: -x["confidence"]):
        code = pred["icd_code"]
        desc = pred.get("description", "")
        conf = pred["confidence"]
        conf_pct = f"{conf * 100:.1f}%"

        # Colour-coded confidence
        if conf >= 0.75:
            conf_color = ACCENT_TEAL
        elif conf >= 0.50:
            conf_color = MED_BLUE
        else:
            conf_color = WARN_AMBER

        table_data.append([
            Paragraph(f"<b>{code}</b>", styles["body"]),
            Paragraph(desc, styles["body"]),
            Paragraph(f'<font color="#{conf_color.hexval()[2:]}">{conf_pct}</font>', styles["body"]),
            Paragraph(
                f'<font color="#{conf_color.hexval()[2:]}">{"█" * int(conf * 20)}{"░" * (20 - int(conf * 20))}</font>',
                ParagraphStyle("bar", fontSize=7, fontName="Helvetica", leading=10),
            ),
        ])

    summary_table = Table(
        table_data,
        colWidths=[28 * mm, 80 * mm, 22 * mm, 47 * mm],
        repeatRows=1,
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#C5D5E8")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 6 * mm))

    # ── Evidence Section ─────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_BLUE))
    story.append(Paragraph("Evidence Snippets (per code)", styles["section_header"]))

    for pred in sorted(predictions, key=lambda x: -x["confidence"]):
        code = pred["icd_code"]
        desc = pred.get("description", "")
        evidence_list = pred.get("evidence", [])
        if not evidence_list:
            continue

        story.append(
            Paragraph(f"<b>{code}</b>  —  {desc}", styles["body"])
        )
        for i, snippet in enumerate(evidence_list[:3], 1):
            story.append(
                Paragraph(f'{i}. "{snippet.strip()}"', styles["evidence"])
            )
        story.append(Spacer(1, 3 * mm))

    # ── Model Info ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        f"Model: {model_info}  ·  Threshold: 0.50  ·  Dataset: MIMIC-IV-Note (top-50 ICD-10-CM codes)",
        styles["small"],
    ))
    story.append(Spacer(1, 4 * mm))

    # ── Disclaimer ───────────────────────────────────────────────────────────
    disclaimer_text = (
        "⚠ DISCLAIMER: This report is AI-generated for research purposes only. "
        "It is NOT a substitute for clinical coding by a certified medical coder or physician. "
        "All codes must be reviewed and verified by qualified clinical staff before use."
    )
    disclaimer_para = Paragraph(disclaimer_text, styles["disclaimer"])
    disclaimer_box = Table(
        [[disclaimer_para]],
        colWidths=[177 * mm],
    )
    disclaimer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), RED_LIGHT),
        ("BOX",        (0, 0), (-1, -1), 1, colors.HexColor("#C0392B")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(disclaimer_box)

    # Build PDF
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    abs_path = os.path.abspath(output_path)
    print(f"✅ Report saved: {abs_path}")
    return abs_path
