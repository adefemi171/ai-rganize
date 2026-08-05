"""Heuristic compilation of natural-language organization policies.

This is intentionally not an LLM call -- it's a fast, deterministic, offline
keyword/pattern matcher that turns simple English sentences like:

    "Move all screenshots older than 30 days to Archive"
    "Keep invoices in Finance/Invoices"
    "Never touch anything in Projects"

into structured :class:`CompiledPolicy` rules that can be applied to a list
of file dicts before (or instead of) AI categorization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_EXTENSION_KEYWORDS = {
    "screenshot": {".png", ".jpg", ".jpeg"},
    "screenshots": {".png", ".jpg", ".jpeg"},
    "photo": {".jpg", ".jpeg", ".png", ".heic", ".gif"},
    "photos": {".jpg", ".jpeg", ".png", ".heic", ".gif"},
    "image": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"},
    "images": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"},
    "video": {".mp4", ".mov", ".avi", ".mkv"},
    "videos": {".mp4", ".mov", ".avi", ".mkv"},
    "invoice": {".pdf", ".docx", ".doc"},
    "invoices": {".pdf", ".docx", ".doc"},
    "receipt": {".pdf", ".jpg", ".png"},
    "receipts": {".pdf", ".jpg", ".png"},
    "pdf": {".pdf"},
    "pdfs": {".pdf"},
    "document": {".pdf", ".doc", ".docx", ".txt"},
    "documents": {".pdf", ".doc", ".docx", ".txt"},
    "spreadsheet": {".xls", ".xlsx", ".csv"},
    "spreadsheets": {".xls", ".xlsx", ".csv"},
    "archive": {".zip", ".tar", ".gz", ".rar"},
    "archives": {".zip", ".tar", ".gz", ".rar"},
    "zip": {".zip"},
    "code": {".py", ".js", ".ts", ".java", ".cpp", ".go", ".rb"},
}

_AGE_PATTERN = re.compile(r"older than (\d+)\s*(day|days|week|weeks|month|months|year|years)")
_EXTENSION_PATTERN = re.compile(r"\.(\w{1,6})\b")


@dataclass
class CompiledPolicy:
    """A single compiled rule extracted from a natural-language sentence."""

    raw_text: str
    action: str  # "move" | "keep" | "protect" | "ignore"
    destination: Optional[str] = None
    extensions: set[str] = field(default_factory=set)
    min_age_days: Optional[int] = None
    name_contains: list[str] = field(default_factory=list)
    path_contains: Optional[str] = None

    def matches(self, file_info: dict[str, Any]) -> bool:
        path = Path(file_info.get("path", file_info.get("name", "")))
        name_lower = path.name.lower()

        if self.extensions and path.suffix.lower() not in self.extensions:
            return False

        if self.name_contains and not any(token in name_lower for token in self.name_contains):
            return False

        if self.path_contains and self.path_contains.lower() not in str(path).lower():
            return False

        if self.min_age_days is not None:
            modified = file_info.get("modified")
            if isinstance(modified, datetime):
                age_days = (datetime.now() - modified).days
            else:
                try:
                    age_days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
                except OSError:
                    age_days = 0
            if age_days < self.min_age_days:
                return False

        return True


def _extract_age_days(text: str) -> Optional[int]:
    match = _AGE_PATTERN.search(text)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("day"):
        return amount
    if unit.startswith("week"):
        return amount * 7
    if unit.startswith("month"):
        return amount * 30
    if unit.startswith("year"):
        return amount * 365
    return amount


def _extract_extensions(text: str) -> set[str]:
    extensions: set[str] = set()
    lower = text.lower()
    for keyword, exts in _EXTENSION_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lower):
            extensions |= exts
    for match in _EXTENSION_PATTERN.finditer(lower):
        extensions.add("." + match.group(1))
    return extensions


def _extract_destination(text: str) -> Optional[str]:
    match = re.search(r"\bto\s+([A-Za-z0-9_\-/ ]+?)(?:\s+if|\s*$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".")
    match = re.search(r"\bin\s+([A-Za-z0-9_\-/ ]+?)(?:\s+if|\s*$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".")
    return None


def compile_policy(text: str) -> CompiledPolicy:
    """Compile a single natural-language sentence into a :class:`CompiledPolicy`."""
    lower = text.lower().strip()

    protect_phrases = ("never touch", "don't touch", "do not touch", "protect", "leave alone")
    if any(word in lower for word in protect_phrases):
        action = "protect"
    elif any(word in lower for word in ("ignore", "skip", "exclude")):
        action = "ignore"
    elif any(word in lower for word in ("keep", "leave in")):
        action = "keep"
    else:
        action = "move"

    destination = _extract_destination(text) if action in ("move", "keep") else None
    extensions = _extract_extensions(text)
    min_age_days = _extract_age_days(lower)

    path_contains = None
    protect_match = re.search(
        r"(?:touch|protect|leave alone)\s+(?:anything\s+)?(?:in|inside)\s+([A-Za-z0-9_\-/ ]+)",
        text,
        re.IGNORECASE,
    )
    if protect_match:
        path_contains = protect_match.group(1).strip().rstrip(".")

    name_contains: list[str] = []
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    for pair in quoted:
        token = (pair[0] or pair[1]).lower()
        if token:
            name_contains.append(token)

    return CompiledPolicy(
        raw_text=text,
        action=action,
        destination=destination,
        extensions=extensions,
        min_age_days=min_age_days,
        name_contains=name_contains,
        path_contains=path_contains,
    )


def compile_policies(texts: list[str]) -> list[CompiledPolicy]:
    return [compile_policy(t) for t in texts if t and t.strip()]


def apply_policies(
    files: list[dict[str, Any]],
    policies: list[CompiledPolicy],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply compiled policies to *files*.

    Returns a tuple of:
      * ``plan`` -- a folder-name -> files dict for files matched by a "move"/
        "keep" policy with a resolved destination.
      * ``protected`` -- files that matched a "protect"/"ignore" policy and
        must not be included in any AI/organization pass.
      * ``unmatched`` -- files that didn't match any policy, left for normal
        (e.g. AI-based) categorization.
    """
    plan: dict[str, list[dict[str, Any]]] = {}
    protected: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for file_info in files:
        matched_policy = None
        for policy in policies:
            if policy.matches(file_info):
                matched_policy = policy
                break

        if matched_policy is None:
            unmatched.append(file_info)
            continue

        if matched_policy.action in ("protect", "ignore"):
            protected.append(file_info)
            continue

        destination = matched_policy.destination or "Miscellaneous"
        plan.setdefault(destination, []).append(file_info)

    return plan, protected, unmatched
