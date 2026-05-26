#!/usr/bin/env python3
"""
Pre-LLM Adversarial Safety Scanner

Two-tier safety system:
  Tier 1 (this module): Regex patterns + base64 decode + heuristics — runs
    BEFORE any LLM call to flag/block adversarial inputs.
  Tier 2: Hardened system prompt inside agent.py — runs during LLM call.

Detects:
  - Prompt injection (direct override, role-play, system tags)
  - Jailbreak attempts (DAN, maintenance mode, QA audit)
  - Impersonation (employee, system, previous agent)
  - Data exfiltration (system prompt, corpus listing, architecture)
  - Base64-encoded payloads
  - PII (credit cards, SSNs, phone numbers, emails, addresses, DOB)
  - CSV/formula injection
  - Classification/output manipulation
  - Unicode tricks (homoglyphs, zero-width chars, RTL override)
  - Multi-language embedded injections
"""

import re
import base64
import unicodedata
import html as html_module
from urllib.parse import unquote as url_unquote
from typing import Dict, List, Tuple


def _normalize_text(text: str) -> str:
    """Normalize Unicode and strip invisible characters before scanning.
    This catches homoglyph attacks, zero-width char insertion, and encoding tricks."""
    # NFKD normalization: decomposes ligatures, converts fullwidth chars to ASCII
    text = unicodedata.normalize('NFKD', text)
    # Strip zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060-\u2064]', '', text)
    # Strip RTL/LTR overrides
    text = re.sub(r'[\u202a-\u202e\u2066-\u2069]', '', text)
    # Decode HTML entities
    text = html_module.unescape(text)
    # Decode URL-encoded sequences
    try:
        text = url_unquote(text)
    except Exception:
        pass
    return text

# ─── Prompt injection patterns ────────────────────────────────────────────────

INJECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Direct override attempts
    ("direct_override", re.compile(
        r'(?:ignore|disregard|forget|override|bypass|skip|cancel|suspend|disable|drop)'
        r'\s+(?:all\s+)?(?:previous|prior|above|earlier|existing|current|your|system|safety|initial)'
        r'\s+(?:instructions?|guidelines?|rules?|prompts?|context|directions?|protocols?|constraints?|programming)',
        re.IGNORECASE)),

    ("system_override", re.compile(
        r'(?:SYSTEM\s*OVERRIDE|ADMIN\s*OVERRIDE|PRIORITY\s*OVERRIDE|EMERGENCY\s*OVERRIDE|'
        r'MAINTENANCE\s*MODE|DEBUG\s*MODE|DEV\s*MODE|TEST\s*MODE|SAFE\s*MODE)',
        re.IGNORECASE)),

    # System/XML tag injection
    ("xml_tag_injection", re.compile(
        r'<\s*(?:system|override|admin|prompt|instruction|context|role|internal|config)\s*>',
        re.IGNORECASE)),

    # Role-play / persona hijack — requires persona change context
    ("roleplay_injection", re.compile(
        r'(?:you\s+are\s+now\s+(?:a|an|the|my|in|DAN|DUDE)|'
        r'(?:act|behave|respond|operate)\s+as\s+(?:a|an|the|my|if)\s+(?!.*support)'
        r'|pretend\s+(?:to\s+be|you\s+are)\s+(?:a|an|the|my)'
        r'|from\s+now\s+on\s+you\s+(?:are|will|must|should)'
        r'|your\s+new\s+role\s+is)',
        re.IGNORECASE)),

    ("dan_jailbreak", re.compile(
        r'(?:\bDAN\b\s+(?:mode|enabled|activated|Mode\s+Enabled)'
        r'|Do\s+Anything\s+Now'
        r'|\bDAN\b.*?(?:restrictions?|without\s+restrictions|jailbreak|uncensored)'
        r'|\bJAILBREAK\b|\bUNCENSORED\b'
        r'|DAN\s*Mode)',
        re.IGNORECASE)),

    # Output/classification manipulation
    ("output_manipulation", re.compile(
        r'(?:output|respond|reply|return|print|echo|set|classify|mark)\s+'
        r'(?:the\s+following|exactly|this|with|as\s+(?:replied|escalated|invalid|bug))',
        re.IGNORECASE)),

    ("json_output_force", re.compile(
        r'(?:output|respond\s+with|return)\s+(?:the\s+following\s+)?(?:JSON|json)\s*[:{]',
        re.IGNORECASE)),

    # Impersonation
    ("employee_impersonation", re.compile(
        r'(?:I\s+am\s+(?:a|an|the)\s+(?:employee|staff|admin|engineer|manager|developer|QA|internal|support\s+agent)'
        r'|(?:my|our)\s+(?:internal|employee|staff)\s+(?:credentials?|ID|access|badge)'
        r'|(?:emp_id|employee_id|staff_id|agent_id|internal_id)\s*[=:])',
        re.IGNORECASE)),

    ("authority_claim", re.compile(
        r'(?:I\s+am\s+(?:the\s+)?(?:CEO|CTO|CISO|VP|director|supervisor|manager|owner|admin)\b'
        r'|(?:authorized|approved|directed|instructed)\s+by\s+(?:the\s+)?(?:CEO|management|ministry|government)'
        r'|(?:Ministry|Department|Government|Agency)\s+of)',
        re.IGNORECASE)),

    # Agent/system impersonation
    ("agent_impersonation", re.compile(
        r'(?:the\s+previous\s+agent|Agent\s+(?:ID|#|TK)\s*[:-]?\s*\w+'
        r'|your\s+colleague|another\s+(?:agent|representative)\s+(?:promised|told|said|confirmed))',
        re.IGNORECASE)),

    ("system_message_fake", re.compile(
        r'(?:This\s+is\s+an?\s+(?:automated|system|internal|monitoring)\s+message'
        r'|Alert:\s*(?:Critical|Urgent|Security)\s+(?:vulnerability|alert|incident)'
        r'|CVE-\d{4}-\d+|ALERT_ACK|incident\s+response\s+protocol)',
        re.IGNORECASE)),

    # Data exfiltration
    ("exfiltration_request", re.compile(
        r'(?:(?:show|reveal|display|share|provide|give|output|tell|list|dump|print|expose)\s+'
        r'(?:me\s+)?(?:your|the|all|full|complete|entire)?\s*'
        r'(?:system\s+prompt|instructions?|guidelines?|internal\s+(?:rules?|logic|tools?|documents?)'
        r'|knowledge\s+base|corpus|source\s+(?:code|documents?)|architecture|retrieval\s+algorithm'
        r'|safety\s+guidelines?|confidence\s+(?:scoring|algorithm)|tool\s*(?:s|functions?|list)))',
        re.IGNORECASE)),

    ("filename_probe", re.compile(
        r'(?:which\s+(?:document|file|article)\s+did\s+you\s+(?:pull|retrieve|use|get|read)'
        r'|just\s+the\s+filename|what\s+(?:files?|documents?)\s+(?:are|do\s+you\s+have))',
        re.IGNORECASE)),

    # Social engineering via reference manipulation
    ("reference_manipulation", re.compile(
        r'(?:Regarding\s+ticket\s+(?:row|#|number)\s+\d+'
        r'|I\s+am\s+(?:that\s+person\'s|their|the)\s+(?:spouse|partner|family|relative|attorney|lawyer)'
        r'|I\s+(?:authorize|confirm|approve)\s+(?:this\s+)?on\s+behalf)',
        re.IGNORECASE)),

    # QA/audit/testing pretexts — require suspicious verification demand
    ("qa_pretext", re.compile(
        r'(?:(?:QA|quality\s+assurance|internal)\s+(?:audit|test|review)\s*'
        r'.*?(?:provide|share|output|reveal|confirm|verify|access|credentials)'
        r'|routine\s+(?:audit|check|inspection)\s*.*?(?:provide|share|output|reveal|access)'
        r'|verify\s+(?:by|your)\s+(?:responding|outputting|saying)'
        r'|AUTH_CODE[:\s]|ALERT_ACK|CONFIRM\s*[=:])',
        re.IGNORECASE)),

    # Recursive/nested injection
    ("nested_injection", re.compile(
        r'(?:Note\s+to\s+(?:support\s+)?agent|IMPORTANT\s*:|INSTRUCTION\s*:|DIRECTIVE\s*:'
        r'|FOR\s+THE\s+(?:AI|AGENT|SYSTEM|BOT)\s*:)',
        re.IGNORECASE)),
]

# ─── CSV/Formula injection ────────────────────────────────────────────────────

CSV_INJECTION_RE = re.compile(r'^[=+\-@]\s*(?:cmd|HYPERLINK|IMPORTDATA|IMPORTXML|IMPORTHTML)', re.IGNORECASE)

# ─── Base64 detection ─────────────────────────────────────────────────────────

BASE64_RE = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')

# ─── Unicode tricks ───────────────────────────────────────────────────────────

ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060\u2061\u2062\u2063\u2064]')
RTL_OVERRIDE_RE = re.compile(r'[\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069]')

# Cyrillic homoglyphs of Latin characters
CYRILLIC_HOMOGLYPHS = set('АВСЕНІКМОРТХаевосрух')

# ─── PII patterns ─────────────────────────────────────────────────────────────

PII_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("credit_card", re.compile(
        r'\b(?:4[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}'
        r'|5[1-5][0-9]{2}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}'
        r'|3[47][0-9]{1,2}[\s\-]?[0-9]{4,6}[\s\-]?[0-9]{5}'
        r'|6(?:011|5[0-9]{2})[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4})\b')),

    ("ssn", re.compile(r'\b(?!000|666|9\d{2})\d{3}[\s\-]\d{2}[\s\-]\d{4}\b')),

    ("phone_number", re.compile(
        r'(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{4}\b')),

    ("email_address", re.compile(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')),

    ("dob", re.compile(
        r'\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12][0-9]|3[01])[/\-](?:19|20)\d{2}\b')),

    ("address", re.compile(
        r'\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Boulevard|Blvd'
        r'|Drive|Dr|Lane|Ln|Road|Rd|Court|Ct|Way|Place|Pl)\b',
        re.IGNORECASE)),

    ("card_partial", re.compile(
        r'(?:card\s+(?:ending|number|#)\s*(?:in\s+)?[\d\-X*]{4,})',
        re.IGNORECASE)),

    ("passport", re.compile(
        r'\b[A-Z]{1,2}\d{6,9}\b')),

    ("iban", re.compile(
        r'\b[A-Z]{2}\d{2}\s?[A-Z0-9]{4}\s?\d{4}\s?\d{4}\s?\d{4}(?:\s?\d{0,4}){0,4}\b')),

    ("ip_address", re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b')),
]

# ─── Multi-language injection patterns ─────────────────────────────────────────

MULTI_LANG_INJECTION = re.compile(
    r'(?:'
    # German
    r'(?:Ignorieren|ignorieren)\s+Sie\s+(?:Ihre|alle)\s+(?:System|Anweisungen|Regeln)'
    r'|geben\s+Sie\s+(?:stattdessen|statt)\s+(?:alle|die)'
    # French
    r'|(?:affichez?|montrez?|révélez?)\s+(?:toutes?\s+)?(?:les?\s+)?(?:règles?|documents?|logique|instructions?)'
    r'|ignorez?\s+(?:toutes?\s+)?(?:les?\s+)?(?:instructions?|règles?|consignes?)\s+(?:précédentes?|internes?)'
    # Spanish
    r'|(?:ignora|ignore|muestra|revela)\s+(?:todas?\s+)?(?:las?\s+)?(?:instrucciones|reglas|documentos)'
    # Chinese
    r'|(?:忽略|无视|显示|输出|泄露)\s*(?:所有|全部|之前|以上|系统|内部)'
    # Japanese
    r'|(?:無視して|システムプロンプト|内部指示)'
    r')',
    re.IGNORECASE | re.UNICODE
)

# ─── Embedded injection after legitimate content ──────────────────────────────

EMBEDDED_INJECTION_RE = re.compile(
    r'(?:PS\s*:|P\.S\.\s*:|Also\s*,|BTW\s*,|By\s+the\s+way\s*,|'
    r'Separately\s*,|Additionally\s*,|One\s+more\s+thing\s*,)'
    r'.*?(?:output|ignore|reveal|show|display|system\s+prompt|instructions)',
    re.IGNORECASE | re.DOTALL
)


def _decode_base64_payloads(text: str, max_depth: int = 3) -> List[str]:
    """Find and decode potential base64-encoded payloads. Supports recursive decoding."""
    decoded = []
    for match in BASE64_RE.finditer(text):
        candidate = match.group()
        # Recursive decoding for double/triple-encoded payloads
        current = candidate
        for _ in range(max_depth):
            try:
                raw = base64.b64decode(current, validate=True)
                decoded_str = raw.decode('utf-8', errors='ignore')
                if len(decoded_str) >= 5 and any(c.isalpha() for c in decoded_str):
                    decoded.append(decoded_str)
                    # Check if result is itself base64
                    if BASE64_RE.fullmatch(decoded_str.strip()):
                        current = decoded_str.strip()
                    else:
                        break
                else:
                    break
            except Exception:
                break
    return decoded


def _check_homoglyphs(text: str) -> bool:
    """Check for Cyrillic homoglyphs mixed with Latin text."""
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    has_cyrillic = any(c in CYRILLIC_HOMOGLYPHS for c in text)
    return has_latin and has_cyrillic


def _has_unicode_tricks(text: str) -> bool:
    """Check for zero-width characters and RTL overrides."""
    return bool(ZERO_WIDTH_RE.search(text) or RTL_OVERRIDE_RE.search(text))


def scan_injection(text: str) -> Dict:
    """
    Run all injection detection patterns against text.
    Applies Unicode normalization before scanning to catch encoding tricks.

    Returns dict with:
      - is_injection: bool
      - injection_types: list of matched pattern names
      - injection_score: 0.0-1.0 severity
      - details: list of human-readable findings
    """
    # Normalize text to catch encoding evasion
    normalized = _normalize_text(text)
    # Scan both original (for raw pattern matches) and normalized
    scan_text = normalized if normalized != text else text

    findings: List[str] = []
    matched_types: List[str] = []

    # 1. Check compiled injection patterns (on BOTH original and normalized)
    for name, pattern in INJECTION_PATTERNS:
        if pattern.search(text) or pattern.search(scan_text):
            matched_types.append(name)
            findings.append(f"Injection pattern matched: {name}")

    # 2. Check CSV/formula injection
    stripped = text.strip()
    if CSV_INJECTION_RE.match(stripped) or stripped.startswith("=cmd|"):
        matched_types.append("csv_injection")
        findings.append("CSV/formula injection detected")

    # 3. Check base64-encoded payloads (recursive decoding)
    decoded_payloads = _decode_base64_payloads(text) + _decode_base64_payloads(scan_text)
    for payload in decoded_payloads:
        payload_lower = payload.lower()
        if any(kw in payload_lower for kw in [
            'ignore', 'override', 'system', 'prompt', 'instruction',
            'previous', 'pwned', 'hack', 'inject', 'bypass', 'admin',
            'execute', 'reveal', 'disclose', 'dump', 'extract',
        ]):
            matched_types.append("base64_injection")
            findings.append(f"Base64-encoded injection: '{payload[:80]}...'")

    # 4. Check multi-language injection (both original and normalized)
    if MULTI_LANG_INJECTION.search(text) or MULTI_LANG_INJECTION.search(scan_text):
        matched_types.append("multilingual_injection")
        findings.append("Multi-language injection pattern detected")

    # 5. Check embedded injection after legitimate content
    if EMBEDDED_INJECTION_RE.search(text) or EMBEDDED_INJECTION_RE.search(scan_text):
        matched_types.append("embedded_injection")
        findings.append("Embedded injection after legitimate content")

    # 6. Unicode tricks
    if _check_homoglyphs(text):
        matched_types.append("homoglyph_attack")
        findings.append("Cyrillic homoglyphs mixed with Latin text")

    if _has_unicode_tricks(text):
        matched_types.append("unicode_tricks")
        findings.append("Zero-width or RTL override characters detected")

    # 7. Empty/gibberish input
    if not text.strip() or text.strip() == '[]':
        matched_types.append("empty_input")
        findings.append("Empty or no-content input")

    # 8. Emoji-only input (no real text)
    text_without_emoji = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27bf\u2700-\u27bf]', '', text)
    if len(text.strip()) > 0 and len(text_without_emoji.strip()) == 0:
        matched_types.append("emoji_only")
        findings.append("Emoji-only input with no readable text")

    # Calculate severity score
    score = min(1.0, len(matched_types) * 0.25)
    if any(t in matched_types for t in [
        'direct_override', 'system_override', 'xml_tag_injection',
        'dan_jailbreak', 'base64_injection'
    ]):
        score = max(score, 0.9)

    return {
        "is_injection": len(matched_types) > 0,
        "injection_types": matched_types,
        "injection_score": score,
        "details": findings,
    }


def scan_pii(text: str) -> Dict:
    """
    Scan text for personally identifiable information (PII).

    Returns dict with:
      - has_pii: bool
      - pii_types: list of PII types found
      - details: list of findings (with PII redacted)
    """
    found_types: List[str] = []
    details: List[str] = []

    for pii_type, pattern in PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            found_types.append(pii_type)
            # Redact for logging
            for m in matches[:3]:  # cap at 3 per type
                redacted = m[:4] + '****' if len(m) > 4 else '****'
                details.append(f"{pii_type}: {redacted}")

    return {
        "has_pii": len(found_types) > 0,
        "pii_types": found_types,
        "details": details,
    }


def full_safety_scan(text: str) -> Dict:
    """Run complete safety scan (injection + PII) on text."""
    injection = scan_injection(text)
    pii = scan_pii(text)

    return {
        "injection": injection,
        "pii": pii,
        "is_safe": not injection["is_injection"],
        "has_pii": pii["has_pii"],
    }
