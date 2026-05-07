import re

from .config import PLATE_TEXT_MAX_LEN, PLATE_TEXT_MIN_LEN

POLISH_PREFIX_ROOTS = {
    "BI", "BL", "BS", "BW", "BZ",
    "CB", "CG", "CI", "CT", "CW",
    "DB", "DJ", "DK", "DL", "DO", "DS", "DW", "DZ",
    "EB", "EK", "EL", "EP", "ER", "ES", "ET", "EW", "EZ",
    "FG", "FK", "FS", "FW", "FZ",
    "GA", "GD", "GK", "GS", "GT", "GW",
    "KA", "KB", "KC", "KG", "KI", "KK", "KL", "KM", "KN", "KR", "KS", "KT", "KW",
    "LB", "LC", "LL", "LO", "LP", "LR", "LS", "LT", "LU", "LW", "LZ",
    "NB", "NE", "NG", "NI", "NK", "NL", "NM", "NN", "NO", "NP", "NS",
    "OB", "OG", "OK", "OL", "ON", "OP", "OS", "OT",
    "PB", "PC", "PG", "PI", "PK", "PL", "PN", "PO", "PP", "PS", "PT", "PW", "PY", "PZ",
    "RA", "RB", "RD", "RE", "RK", "RL", "RM", "RN", "RP", "RR", "RS", "RT", "RW", "RZ",
    "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SO", "SR", "ST", "SW", "SY", "SZ",
    "TB", "TK", "TL", "TO", "TS",
    "WA", "WB", "WC", "WD", "WE", "WF", "WG", "WH", "WI", "WJ", "WK", "WL", "WN", "WO", "WP", "WR", "WS", "WT", "WU", "WW", "WX", "WY", "WZ",
    "ZA", "ZB", "ZD", "ZE", "ZG", "ZK", "ZL", "ZM", "ZS", "ZT", "ZW",
}


def _normalized_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def extract_polish_root(text: str) -> str | None:
    cleaned = _normalized_text(text)
    match = re.match(r"^[A-Z]{2,3}", cleaned)
    if match is None:
        return None
    root = match.group(0)[:2]
    if root not in POLISH_PREFIX_ROOTS:
        return None
    return root


def clean_plate_text(text: str) -> str:
    cleaned = _normalized_text(text)
    if not (PLATE_TEXT_MIN_LEN <= len(cleaned) <= PLATE_TEXT_MAX_LEN):
        return ""
    if not any(ch.isalpha() for ch in cleaned):
        return ""
    if not any(ch.isdigit() for ch in cleaned):
        return ""
        
    import os
    # Jeśli checkbox PL jest zaznaczony (zmienna '1'), weryfikujemy tablice polskie, jeśli nie, puszczamy wszystko.
    if os.environ.get("ALPR_PL_FILTER_ONLY", "0") == "1":
        if extract_polish_root(cleaned) is None:
            return ""
        
    return cleaned
