"""
Invoice DocAI — shared utilities (v2, improved).

Changes vs v1:
- normalize_date: supports DD/MM/YY, month names ("15 Jan 21"), return_raw option
- normalize_total: return_raw option, negative value validation
- build_manifest: rel_to for portable paths, gt_date_raw / gt_total_raw columns
- compute_field_metrics: ignore_empty_gt, support count
- evaluate: ignore_empty_gt (default True for fair evaluation)
- NEW: extract_total_from_lines — keyword-priority + range filter (fixes phone-number bug)
- NEW: extract_vendor_from_lines — skip-pattern filtering
- NEW: extract_date_from_lines, extract_fields_from_lines
- NEW: messenger_corrupt for robustness evaluation
"""

import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Extended date patterns including 2-digit year format (DD/MM/YY)
DATE_PATTERNS = [
    re.compile(r"\b(\d{2})[./-](\d{2})[./-](\d{4})\b"),  # DD/MM/YYYY or MM/DD/YYYY
    re.compile(r"\b(\d{4})[./-](\d{2})[./-](\d{2})\b"),  # YYYY-MM-DD
    re.compile(r"\b(\d{2})[./-](\d{2})[./-](\d{2})\b"),  # DD/MM/YY or MM/DD/YY
]

# Month name patterns (e.g., "15 Jan 21")
MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
MONTH_NAME_PATTERN = re.compile(
    r"\b(\d{1,2})\s+([a-z]{3,9})\s+(\d{2,4})\b", re.IGNORECASE
)

MONEY_PATTERN = re.compile(r"[-+]?\d+[\d\s.,]*\d")

DATE_REGEX = re.compile(
    r"\b(\d{2}[./-]\d{2}[./-]\d{4}|\d{4}[./-]\d{2}[./-]\d{2})\b"
)

# Lines to skip when looking for vendor name
_VENDOR_SKIP_PATTERNS = [
    re.compile(r"\d{2}[./-]\d{2}[./-]\d{4}", re.IGNORECASE),
    re.compile(r"\b(tel|fax|phone|email|www|http)\b", re.IGNORECASE),
    re.compile(r"\b(no\.\s*\d|no:\s*\d)", re.IGNORECASE),
    re.compile(r"\b(gst\s*id|tax\s*id|reg\s*no)\b", re.IGNORECASE),
    re.compile(r"\b(cashier|counter|receipt|invoice\s*no)\b", re.IGNORECASE),
]

# Keywords that disqualify a line from being "TOTAL"
_TOTAL_SKIP_KEYWORDS = [
    "subtotal", "sub total", "sub-total",
    "total qty", "total item", "total quantity",
    "total excluded", "total tax",
]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_total(value: Any, return_raw: bool = False) -> str:
    """Normalize monetary value. Returns "X.XX" or "".
    If return_raw=True, returns tuple (normalized, raw_text).
    """
    if value is None:
        return ("", "") if return_raw else ""
    text = str(value).strip()
    if not text:
        return ("", "") if return_raw else ""

    raw_text = text
    text = text.replace(" ", "").replace("$", "").replace("€", "").replace("RM", "")
    text = text.replace(",", ".")
    match = MONEY_PATTERN.search(text)
    if not match:
        return ("", raw_text) if return_raw else ""
    candidate = match.group(0)
    if candidate.count(".") > 1:
        parts = candidate.split(".")
        candidate = "".join(parts[:-1]) + "." + parts[-1]
    try:
        value_dec = Decimal(candidate)
        if value_dec < 0:
            return ("", raw_text) if return_raw else ""
        result = f"{value_dec:.2f}"
        return (result, raw_text) if return_raw else result
    except InvalidOperation:
        return ("", raw_text) if return_raw else ""


def normalize_date(value: Any, return_raw: bool = False) -> str:
    """Normalize date to YYYY-MM-DD format.
    Supports DD/MM/YYYY, YYYY-MM-DD, DD/MM/YY, '15 Jan 21', etc.
    If return_raw=True, returns tuple (normalized, raw_text).
    """
    if value is None:
        return ("", "") if return_raw else ""
    text = str(value).strip()
    if not text:
        return ("", "") if return_raw else ""

    raw_text = text

    # Try month name format first (e.g., "15 Jan 21")
    month_match = MONTH_NAME_PATTERN.search(text)
    if month_match:
        day_str, month_str, year_str = month_match.groups()
        month_lower = month_str.lower()
        if month_lower in MONTH_NAMES:
            try:
                day = int(day_str)
                month = MONTH_NAMES[month_lower]
                year = int(year_str)
                if year < 100:
                    year = 2000 + year if year <= 50 else 1900 + year
                if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                    result = f"{year:04d}-{month:02d}-{day:02d}"
                    return (result, raw_text) if return_raw else result
            except ValueError:
                pass

    # Try numeric date patterns
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        a, b, c = match.groups()

        if len(a) == 4:
            year, month, day = a, b, c
        elif len(c) == 4:
            day, month, year = a, b, c
        elif len(c) == 2:
            day, month, year = a, b, c
            year_int = int(year)
            year = str(2000 + year_int if year_int <= 50 else 1900 + year_int)
        else:
            continue

        try:
            year_int = int(year)
            month_int = int(month)
            day_int = int(day)
            if not (1 <= month_int <= 12 and 1 <= day_int <= 31 and 1900 <= year_int <= 2100):
                continue
            result = f"{year_int:04d}-{month_int:02d}-{day_int:02d}"
            return (result, raw_text) if return_raw else result
        except ValueError:
            continue

    return ("", raw_text) if return_raw else ""


def normalize_vendor(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Field extraction from OCR lines (v2 — improved)
# ---------------------------------------------------------------------------

def extract_total_from_lines(lines: List[str]) -> str:
    """Extract total amount from OCR lines.

    Fixes the v1 bug where phone numbers / registration IDs were picked as
    the 'total' because naive logic chose the largest number.

    Strategy hierarchy:
      1. Lines containing keyword 'TOTAL' (excluding 'SUBTOTAL' etc.) — range-filtered
      2. Lines containing 'AMOUNT' / 'ROUNDING' / 'CHANGE' — range-filtered
      3. Bottom-half numbers with exactly 2 decimal places (monetary format)
      4. Any number with 2 decimal places in the full document
    """
    cleaned = [ln.strip() for ln in lines if ln and ln.strip()]
    if not cleaned:
        return ""

    # --- Strategy 1: lines with keyword "TOTAL" --------------------------
    keyword_candidates: List[Tuple[float, str]] = []
    for ln in cleaned:
        ln_lower = ln.lower()
        if any(skip in ln_lower for skip in _TOTAL_SKIP_KEYWORDS):
            continue
        if not re.search(r"\btotal\b", ln_lower):
            continue
        amount = normalize_total(ln)
        if amount:
            val = float(amount)
            if 0.01 <= val <= 50_000:
                keyword_candidates.append((val, amount))

    if keyword_candidates:
        return keyword_candidates[-1][1]

    # --- Strategy 2: lines with "AMOUNT" / "ROUNDING" / "CHANGE" ----------
    for ln in cleaned:
        ln_lower = ln.lower()
        if re.search(r"\b(amount|rounding|round\s*adj|change)\b", ln_lower):
            amount = normalize_total(ln)
            if amount:
                val = float(amount)
                if 0.01 <= val <= 50_000:
                    return amount

    # --- Strategy 3: bottom-half, numbers with 2 decimal places -----------
    bottom_half = cleaned[len(cleaned) // 2:]
    money_candidates: List[Tuple[float, str]] = []
    for ln in bottom_half:
        matches = re.findall(r"\b(\d{1,6}\.\d{2})\b", ln)
        for m in matches:
            val = float(m)
            if 0.01 <= val <= 50_000:
                money_candidates.append((val, normalize_total(m)))

    if money_candidates:
        money_candidates.sort(key=lambda x: x[0], reverse=True)
        return money_candidates[0][1]

    # --- Strategy 4: any number with 2 decimal places (full doc) ----------
    for ln in cleaned:
        matches = re.findall(r"\b(\d{1,6}\.\d{2})\b", ln)
        for m in matches:
            val = float(m)
            if 0.01 <= val <= 50_000:
                return normalize_total(m)

    return ""


def extract_vendor_from_lines(lines: List[str]) -> str:
    """Extract vendor/company name from the first few OCR lines."""
    cleaned = [ln.strip() for ln in lines if ln and ln.strip()]
    if not cleaned:
        return ""

    candidates: List[Tuple[int, str, int]] = []
    for i, ln in enumerate(cleaned[:6]):
        letters = sum(ch.isalpha() for ch in ln)
        digits = sum(ch.isdigit() for ch in ln)
        if letters < 3:
            continue
        if digits > letters:
            continue
        if any(pat.search(ln) for pat in _VENDOR_SKIP_PATTERNS):
            continue
        candidates.append((i, ln, letters))

    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[0][1]

    for ln in cleaned[:5]:
        if sum(ch.isalpha() for ch in ln) >= 3:
            return ln
    return ""


def extract_date_from_lines(lines: List[str]) -> str:
    """Extract the first date from OCR lines."""
    for ln in lines:
        if not ln:
            continue
        m = DATE_REGEX.search(ln.strip())
        if m:
            d = normalize_date(m.group(1))
            if d:
                return d
    return ""


def extract_fields_from_lines(lines: List[str]) -> Dict[str, str]:
    """Extract vendor, date, total from OCR lines (combined wrapper)."""
    return {
        "pred_vendor": extract_vendor_from_lines(lines),
        "pred_date": extract_date_from_lines(lines),
        "pred_total": extract_total_from_lines(lines),
    }


# ---------------------------------------------------------------------------
# Messenger-grade image corruption
# ---------------------------------------------------------------------------

def messenger_corrupt(
    img_bgr: np.ndarray,
    jpeg_quality: int = 20,
    blur_kernel: int = 5,
    perspective: float = 0.08,
    downscale: float = 0.55,
) -> np.ndarray:
    """Apply messenger-grade corruptions: perspective warp, blur, downscale, JPEG."""
    import cv2

    h, w = img_bgr.shape[:2]

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dx = max(1, int(w * perspective))
    dy = max(1, int(h * perspective))
    dst = np.float32([
        [random.randint(0, dx), random.randint(0, dy)],
        [w - 1 - random.randint(0, dx), random.randint(0, dy)],
        [w - 1 - random.randint(0, dx), h - 1 - random.randint(0, dy)],
        [random.randint(0, dx), h - 1 - random.randint(0, dy)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    k = max(3, blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1)
    blurred = cv2.GaussianBlur(warped, (k, k), 0)

    new_w, new_h = max(1, int(w * downscale)), max(1, int(h * downscale))
    small = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    ok, enc = cv2.imencode(".jpg", restored, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        return restored
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# SROIE label parsing & manifest building
# ---------------------------------------------------------------------------

def parse_sroie_label(label_path: Path) -> Dict[str, str]:
    content = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    result = {"vendor": "", "date": "", "total": ""}
    if not content:
        return result

    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            result["vendor"] = str(payload.get("company") or payload.get("vendor") or "")
            result["date"] = str(payload.get("date") or "")
            result["total"] = str(payload.get("total") or payload.get("amount") or "")
            return result
    except json.JSONDecodeError:
        pass

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    lowered = [line.lower() for line in lines]

    for line in lines:
        if re.search(r"\btotal\b", line, flags=re.IGNORECASE):
            amount = normalize_total(line)
            if amount:
                result["total"] = amount
    if not result["total"]:
        amounts = [normalize_total(line) for line in lines]
        amounts = [a for a in amounts if a]
        if amounts:
            result["total"] = max(amounts, key=lambda x: Decimal(x))

    for line in lines:
        d = normalize_date(line)
        if d:
            result["date"] = d
            break

    if lines:
        result["vendor"] = lines[0]
    for idx, line in enumerate(lowered):
        if "company" in line or "vendor" in line:
            value = lines[idx].split(":", 1)[-1].strip()
            result["vendor"] = value or result["vendor"]
            break

    return result


def build_manifest(
    images_dir: Path,
    labels_dir: Path,
    rel_to: Optional[Path] = None,
) -> pd.DataFrame:
    """Build manifest with image/label paths and ground truth fields.

    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing label files.
        rel_to: If provided, paths are stored relative to this directory.

    Returns:
        DataFrame with columns: id, image_path, label_path, gt_vendor,
        gt_date, gt_total, gt_date_raw, gt_total_raw
    """
    image_paths = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpeg"))
    )
    records: List[Dict[str, str]] = []

    for image_path in image_paths:
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"
        gt = parse_sroie_label(label_path) if label_path.exists() else {
            "vendor": "", "date": "", "total": ""
        }

        date_normalized, date_raw = normalize_date(gt["date"], return_raw=True)
        total_normalized, total_raw = normalize_total(gt["total"], return_raw=True)

        if rel_to:
            try:
                img_path_str = image_path.relative_to(rel_to).as_posix()
                lbl_path_str = (
                    label_path.relative_to(rel_to).as_posix()
                    if label_path.exists() else ""
                )
            except ValueError:
                img_path_str = str(image_path)
                lbl_path_str = str(label_path) if label_path.exists() else ""
        else:
            img_path_str = str(image_path)
            lbl_path_str = str(label_path) if label_path.exists() else ""

        records.append({
            "id": stem,
            "image_path": img_path_str,
            "label_path": lbl_path_str,
            "gt_vendor": gt["vendor"],
            "gt_date": date_normalized,
            "gt_total": total_normalized,
            "gt_date_raw": date_raw,
            "gt_total_raw": total_raw,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_field_metrics(
    y_true: List[str],
    y_pred: List[str],
    ignore_empty_gt: bool = False,
) -> Dict[str, float]:
    """Compute precision, recall, F1 for a single field.

    Args:
        ignore_empty_gt: If True, skip samples where GT is empty.
    Returns:
        Dict with keys: precision, recall, f1, tp, fp, fn, support
    """
    tp = fp = fn = 0
    support = 0

    for t, p in zip(y_true, y_pred):
        t_strip = str(t).strip()
        p_strip = str(p).strip()
        t_ok = bool(t_strip)
        p_ok = bool(p_strip)

        if ignore_empty_gt and not t_ok:
            continue

        support += 1

        if p_ok and t_ok and p_strip == t_strip:
            tp += 1
        elif p_ok and (not t_ok or p_strip != t_strip):
            fp += 1
        elif t_ok and not p_ok:
            fn += 1
        elif t_ok and p_ok and p_strip != t_strip:
            fp += 1
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "support": support,
    }


def evaluate(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    ignore_empty_gt: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate predictions against ground truth.

    Args:
        ignore_empty_gt: If True, compute metrics only on samples with
                         non-empty GT (recommended for fair evaluation).
    Returns:
        Tuple of (metrics_df, errors_df).
    """
    merged = ground_truth_df.merge(predictions_df, on="id", how="inner")

    for col in ["gt_vendor", "pred_vendor"]:
        merged[col] = merged[col].fillna("").map(normalize_vendor)
    for col in ["gt_date", "pred_date"]:
        merged[col] = merged[col].fillna("").map(normalize_date)
    for col in ["gt_total", "pred_total"]:
        merged[col] = merged[col].fillna("").map(normalize_total)

    metrics_rows = []
    total_tp = total_fp = total_fn = 0
    total_support = 0

    for field in ["vendor", "date", "total"]:
        field_metrics = compute_field_metrics(
            merged[f"gt_{field}"].tolist(),
            merged[f"pred_{field}"].tolist(),
            ignore_empty_gt=ignore_empty_gt,
        )

        if ignore_empty_gt:
            mask = merged[f"gt_{field}"].str.strip() != ""
            subset = merged[mask]
            exact = (
                (subset[f"gt_{field}"] == subset[f"pred_{field}"]).mean()
                if len(subset) else 0.0
            )
        else:
            exact = (
                (merged[f"gt_{field}"] == merged[f"pred_{field}"]).mean()
                if len(merged) else 0.0
            )

        field_metrics["field"] = field
        field_metrics["exact_match"] = exact
        metrics_rows.append(field_metrics)
        total_tp += field_metrics["tp"]
        total_fp += field_metrics["fp"]
        total_fn += field_metrics["fn"]
        total_support += field_metrics["support"]

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) else 0.0
    )

    metrics_rows.append({
        "field": "micro",
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f1,
        "exact_match": np.nan,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "support": total_support,
    })

    metrics_df = pd.DataFrame(metrics_rows)

    errors = merged[
        (merged["gt_vendor"] != merged["pred_vendor"])
        | (merged["gt_date"] != merged["pred_date"])
        | (merged["gt_total"] != merged["pred_total"])
    ].copy()
    errors["num_wrong_fields"] = (
        (errors["gt_vendor"] != errors["pred_vendor"]).astype(int)
        + (errors["gt_date"] != errors["pred_date"]).astype(int)
        + (errors["gt_total"] != errors["pred_total"]).astype(int)
    )
    errors = errors.sort_values("num_wrong_fields", ascending=False)
    return metrics_df, errors
