"""Automated document metadata extraction and taxonomy classification for RAG.

Provides senior engineering, high-precision structural extraction of document
titles, legal taxonomies, execution dates, party identifiers, and scripts directly
from raw OCR or native text prior to database indexing.
"""
from __future__ import annotations

import re
import os
from typing import Any, Dict, List, Optional, Tuple
from core.text_normalizer import has_tamil, is_predominantly_tamil, _TAMIL_RE

# ── Taxonomy Definitions ───────────────────────────────────────────────────

#: Domain classification mapping: category name -> (required weight threshold, list of keyword regexes/phrases)
_TAXONOMY_MAP: Dict[str, List[str]] = {
    "Sale Deed": [
        r"\bsale\s+deed\b", r"\bsold\s+and\s+conveyed\b", r"\babsolute\s+sale\b",
        r"\bvendor\b", r"\bvendee\b", r"\bsale\s+consideration\b", r"கிரையப்\s+பத்திரம்",
    ],
    "Lease / Rental Agreement": [
        r"\blease\s+deed\b", r"\brental\s+agreement\b", r"\ble(?:ncy|ssee)\b", r"\ble(?:ssor|see)\b",
        r"\blessor\b", r"\blessee\b", r"\bmonthly\s+rent\b", r"\btenancy\b", r"குத்தகை", r"வாடகை\s+ஒப்பந்தம்",
    ],
    "Power of Attorney": [
        r"\bpower\s+of\s+attorney\b", r"\bgeneral\s+power\s+of\s+attorney\b", r"\bspecial\s+power\s+of\s+attorney\b",
        r"\bgpa\b", r"\battorney[\s-]+in[\s-]+fact\b", r"பொது\s+அதிகാര\s+பத்திரம்",
    ],
    "Partition Deed": [
        r"\bpartition\s+deed\b", r"\bfamily\s+partition\b", r"\bjoint\s+family\s+property\b",
        r"\bshare\s+in\s+property\b", r"பாகப்\s+பிரிவினை",
    ],
    "Mortgage / Gift / Settlement Deed": [
        r"\bmortgage\s+deed\b", r"\bgift\s+deed\b", r"\bsettlement\s+deed\b",
        r"\brelease\s+deed\b", r"\bhypothecation\b", r"தானப்\s+பத்திரம்", r"அடமானப்\s+பத்திரம்",
    ],
    "Will / Testament": [
        r"\blast\s+will\b", r"\btestament\b", r"\btestator\b", r"\btestatrix\b",
        r"\bbequeath\b", r"\bprobate\b", r"உயில்",
    ],
    "Encumbrance Certificate (EC)": [
        r"\bencumbrance\s+certificate\b", r"\bform\s+no\.?\s*15\b", r"\bform\s+no\.?\s*16\b",
        r"\bsub[\s-]+registrar\b", r"வில்லங்க\s+சான்றிதழ்",
    ],
    "Land / Revenue Record (Patta/Chitta)": [
        r"\bpatta\b", r"\bchitta\b", r"\badangal\b", r"\bfmb\b", r"\bfield\s+measurement\s+book\b",
        r"\bvillage\s+account\b", r"பட்டா", r"சிட்டா", r"அடங்கல்",
    ],
    "Court Order / Legal Pleading": [
        r"\bin\s+the\s+(?:high\s+)?court\s+of\b", r"\bwrit\s+petition\b", r"\bcourt\s+order\b",
        r"\bplaintiff\b", r"\bdefendant\b", r"\bpetitioner\b", r"\brespondent\b", r"\baffidavit\b", r"\bjudgment\b",
    ],
    "Commercial Contract / Agreement": [
        r"\bmemorandum\s+of\s+understanding\b", r"\bservice\s+agreement\b", r"\bnon[\s-]+disclosure\s+agreement\b",
        r"\bcovenant\b", r"\bindemnity\b", r"\bargitration\s+clause\b",
    ],
    "Resume / Identity / Profile": [
        r"\bcurriculum\s+vitae\b", r"\bwork\s+experience\b", r"\btechnical\s+skills\b",
        r"\beducation\b", r"\bacademic\s+background\b", r"\bcertifications\b",
    ],
}

#: Regex for extracting standardized date formats.
_DATE_PATTERNS_RE = re.compile(
    r"\b(?:"
    # DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, YYYY/MM/DD
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    # 25th day of October, 2023 or October 25, 2023 or 25 Oct 2023
    r"(?:1st|2nd|3rd|[0-9]{1,2}(?:th|st|nd|rd)?)\s+(?:day\s+of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:th|st|nd|rd)?[\s,]+\d{2,4}"
    r")\b",
    re.IGNORECASE,
)

#: Common party entity extraction heuristics in legal deeds.
_PARTY_RE = re.compile(
    r"(?:(?:by\s+and\s+between|between|by)|(?:vendor|vendee|lessor|lessee|grantor|grantee|petitioner|respondent|testator|executant)s?\s*(?::|-|hereinafter\s+called|hereinafter\s+referred\s+to\s+as\s+(?:the\s+)?[a-z]+)?)\s*"
    r"([A-Z][A-Za-z\s,.\u0B80-\u0BFF]{3,45}?)(?=\s*(?:,|\bresiding\b|\bt/v/d/o\b|\bd/o\b|\bc/o\b|\bs/o\b|\bw/o\b|\baged\b|\baged\s+about\b|\bp\.?a\.?n\b|and\b|\.|\n))",
    re.IGNORECASE,
)

#: Title exclusion list (common non-title headers).
_TITLE_BLOCKLIST = frozenset({
    "page", "scanned by camscanner", "confidential", "draft", "copy",
    "true copy", "attested copy", "government of tamil nadu",
    "registration department", "table of contents",
})


def extract_title(text: str, filename: Optional[str] = None) -> str:
    """Intelligently extract document title from initial lines or filename fallback.

    Scans the earliest prominent lines for formal document titles (such as
    Deed designations or high-confidence caps titles), falling back to clean
    filename formatting when needed.
    """
    if text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        # Examine the first 25 non-empty lines for prominent titles
        for line in lines[:25]:
            lower = line.lower()
            if any(blk in lower for blk in _TITLE_BLOCKLIST) and len(line) < 40:
                continue
            # Skip pure numbers, dates, or tiny fragments
            if len(line) < 4 or len(line) > 120 or line.replace(".", "").replace("-", "").isdigit():
                continue

            # Strong indicator: matches one of our primary taxonomy headers
            if any(term in lower for term in ["deed", "agreement", "power of attorney", "certificate", "will", "patta", "court", "affidavit", "petition", "பத்திரம்", "ஒப்பந்தம்"]):
                return line[:120].strip()

            # Strong indicator: standalone all-caps line of reasonable length (e.g. "MEMORANDUM OF UNDERSTANDING")
            if line.isupper() and len(line) >= 6 and len(line.split()) <= 12 and not any(ch.isdigit() for ch in line):
                return line[:120].strip()

        # If no explicit keyword or all-caps header matched, take the very first meaningful alphabetical line under 80 chars
        for line in lines[:10]:
            if len(line) >= 6 and len(line) <= 80 and any(c.isalpha() for c in line) and not any(blk in line.lower() for blk in _TITLE_BLOCKLIST):
                return line.strip()

    # Fallback to clean filename
    if filename:
        base = os.path.splitext(os.path.basename(filename))[0]
        # Convert underscores and hyphens to spaces and title case (e.g. "sale_deed_doc_1" -> "Sale Deed Doc 1")
        clean = re.sub(r"[_\-]+", " ", base).strip().title()
        if clean:
            return clean

    return "Untitled Document"


def classify_document(text: str) -> str:
    """Classify document into domain legal/commercial taxonomies via keyword density."""
    if not text:
        return "General Document"

    lower = text.lower()
    scores: Dict[str, int] = {}

    for category, patterns in _TAXONOMY_MAP.items():
        count = 0
        for pat in patterns:
            if re.search(pat, lower):
                count += 1
        if count > 0:
            scores[category] = count

    if not scores:
        if has_tamil(text) or any(t in lower for t in ["schedule", "survey", "sy. no", "taluk", "village", "reg. no"]):
            return "General Legal Document"
        return "General Document"

    # Select category with highest keyword frequency match
    best_category = max(scores.items(), key=lambda item: (item[1], len(item[0])))[0]
    return best_category


def extract_dates(text: str) -> List[str]:
    """Extract and deduplicate execution or effective dates from document text."""
    if not text:
        return []

    matches = _DATE_PATTERNS_RE.findall(text)
    deduped: List[str] = []
    seen: set = set()

    for m in matches:
        clean = " ".join(m.strip().split())
        key = clean.lower()
        if len(clean) >= 6 and key not in seen:
            seen.add(key)
            deduped.append(clean)
        if len(deduped) >= 12:  # Cap at top 12 dates to avoid index inflation
            break

    return deduped


def detect_document_language(text: str) -> str:
    """Detect linguistic distribution of document text.

    Returns:
        - ``"ta"``: Predominantly Tamil (≥ 60% Tamil letters)
        - ``"bilingual"``: Significant mixed Tamil and English/Latin (≥ 10% Tamil, ≥ 15% Latin)
        - ``"en"``: Predominantly English / Latin
    """
    if not text:
        return "en"

    letters = [ch for ch in text if ch.isalpha() or _TAMIL_RE.match(ch)]
    total = len(letters)
    if total == 0:
        return "en"

    tamil_chars = sum(1 for ch in text if _TAMIL_RE.match(ch))
    latin_chars = sum(1 for ch in text if ch.isascii() and ch.isalpha())

    tamil_ratio = tamil_chars / total
    latin_ratio = latin_chars / total

    if tamil_ratio >= 0.60:
        return "ta"
    if tamil_ratio >= 0.10 and latin_ratio >= 0.10:
        return "bilingual"
    return "en"


def extract_parties(text: str) -> Dict[str, List[str]]:
    """Identify key legal entities and party identifiers (Vendors, Lessees, etc.)."""
    if not text:
        return {"parties": []}

    # Restrict party extraction to the first 4000 characters (preambles / execution blocks)
    header_sample = text[:4000]
    matches = _PARTY_RE.findall(header_sample)

    parties: List[str] = []
    seen: set = set()

    for name in matches:
        clean = " ".join(name.strip().split())
        key = clean.lower()
        # Ensure name isn't generic boilerplates or tiny fragments
        if len(clean) >= 3 and key not in seen and not any(w in key for w in ["where", "whereas", "this deed", "office", "registrar"]):
            seen.add(key)
            parties.append(clean)
        if len(parties) >= 8:
            break

    return {"parties": parties}


def extract_document_metadata(text: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """Execute complete automated metadata enrichment suite on document content.

    Returns a high-signal dictionary containing title, taxonomy classification,
    dates, detected language, and party identifiers ready for persistence.
    """
    if not text:
        text = ""

    title = extract_title(text, filename)
    classification = classify_document(text)
    dates = extract_dates(text)
    language = detect_document_language(text)
    parties_info = extract_parties(text)

    return {
        "extracted_title": title,
        "doc_classification": classification,
        "detected_language": language,
        "extracted_dates": dates,
        "primary_date": dates[0] if dates else None,
        "parties": parties_info.get("parties", []),
    }
