"""
pdf_parser.py
─────────────
Extract and anonymise text from discharge summary PDFs.

Dependencies:
    pip install pymupdf presidio-analyzer presidio-anonymizer spacy
    python -m spacy download en_core_web_lg
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedNote:
    """Holds extracted and anonymised note data."""
    raw_text: str
    clean_text: str
    patient_name: str = "Unknown"
    patient_id: str = "Unknown"
    admission_date: str = ""
    discharge_date: str = ""
    page_count: int = 0
    sections: Dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    """
    Extract all text from a PDF using PyMuPDF.

    Returns:
        (full_text, page_count)
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages), len(pages)


# ─────────────────────────────────────────────────────────────────────────────
# Section detection
# ─────────────────────────────────────────────────────────────────────────────

# Common discharge summary section headers (case-insensitive)
SECTION_HEADERS = [
    "chief complaint",
    "history of present illness",
    "past medical history",
    "medications on admission",
    "allergies",
    "physical exam",
    "pertinent results",
    "assessment and plan",
    "discharge diagnosis",
    "discharge medications",
    "discharge condition",
    "discharge instructions",
    "follow-up",
]

def extract_sections(text: str) -> Dict[str, str]:
    """
    Detect and extract sections from a clinical note.
    Returns dict: {section_header: section_text}
    """
    sections = {}
    pattern = "|".join(re.escape(h) for h in SECTION_HEADERS)
    regex = re.compile(rf"(?i)({pattern})\s*[:\-]?\s*\n", re.MULTILINE)

    headers_found = [(m.start(), m.group(1).lower()) for m in regex.finditer(text)]

    for i, (start, header) in enumerate(headers_found):
        end = headers_found[i + 1][0] if i + 1 < len(headers_found) else len(text)
        content = text[start:end].strip()
        sections[header] = content

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Patient info extraction (heuristic — before PHI removal)
# ─────────────────────────────────────────────────────────────────────────────

def extract_patient_info(text: str) -> Tuple[str, str, str, str]:
    """
    Heuristically extract patient name, ID, admission date, discharge date.
    These are preserved in the output report.

    Returns: (name, patient_id, admission_date, discharge_date)
    """
    name = "Unknown"
    patient_id = "Unknown"
    admission_date = ""
    discharge_date = ""

    # Patient name patterns: "Patient Name: John Doe" or "Name: John Doe"
    name_match = re.search(
        r"(?:patient[\s_]?name|name)\s*[:\-]\s*([A-Za-z ,'-]+)", text, re.IGNORECASE
    )
    if name_match:
        name = name_match.group(1).strip().title()

    # MRN / Patient ID
    id_match = re.search(
        r"(?:mrn|patient[\s_]?id|admission[\s_]?id|id)\s*[:\-#]\s*([A-Za-z0-9\-]+)",
        text, re.IGNORECASE
    )
    if id_match:
        patient_id = id_match.group(1).strip()

    # Admission date
    adm_match = re.search(
        r"(?:admission|admitted|admit)[\s_]?(?:date)?\s*[:\-]\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
        text, re.IGNORECASE
    )
    if adm_match:
        admission_date = adm_match.group(1)

    # Discharge date
    dis_match = re.search(
        r"(?:discharge)[\s_]?(?:date)?\s*[:\-]\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
        text, re.IGNORECASE
    )
    if dis_match:
        discharge_date = dis_match.group(1)

    return name, patient_id, admission_date, discharge_date


# ─────────────────────────────────────────────────────────────────────────────
# PHI anonymisation via Microsoft Presidio
# ─────────────────────────────────────────────────────────────────────────────

def anonymise_phi(text: str, keep_entities: Optional[List[str]] = None) -> str:
    """
    Remove Protected Health Information (PHI) using Microsoft Presidio.

    Args:
        text:           Raw clinical note text
        keep_entities:  List of entity types to NOT redact (e.g. ["DATE_TIME"])

    Returns:
        Anonymised text with PHI replaced by [REDACTED]
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError:
        print("⚠ Presidio not installed — returning text unchanged. "
              "Run: pip install presidio-analyzer presidio-anonymizer")
        return text

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    keep = set(keep_entities or [])

    # Entities to detect and redact
    entities = [
        "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN",
        "CREDIT_CARD", "IP_ADDRESS", "LOCATION", "DATE_TIME",
        "MEDICAL_LICENSE", "URL", "NRP",
    ]
    entities = [e for e in entities if e not in keep]

    results = analyzer.analyze(text=text, entities=entities, language="en")

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})},
    )
    return anonymized.text


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_discharge_pdf(
    pdf_path: str,
    anonymise: bool = True,
    keep_dates: bool = True,
) -> ParsedNote:
    """
    Full pipeline: extract PDF text → extract patient info → anonymise PHI.

    Args:
        pdf_path:    Path to discharge summary PDF
        anonymise:   Whether to apply Presidio PHI removal (default True)
        keep_dates:  Whether to preserve DATE_TIME entities (useful for clinical context)

    Returns:
        ParsedNote dataclass
    """
    raw_text, page_count = extract_text_from_pdf(pdf_path)

    # Extract patient info BEFORE anonymisation
    name, patient_id, admission_date, discharge_date = extract_patient_info(raw_text)

    # Anonymise
    keep_entities = ["DATE_TIME"] if keep_dates else []
    clean_text = anonymise_phi(raw_text, keep_entities=keep_entities) if anonymise else raw_text

    # Light text normalisation
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # Section extraction (on clean text)
    sections = extract_sections(clean_text)

    return ParsedNote(
        raw_text=raw_text,
        clean_text=clean_text,
        patient_name=name,
        patient_id=patient_id,
        admission_date=admission_date,
        discharge_date=discharge_date,
        page_count=page_count,
        sections=sections,
    )
