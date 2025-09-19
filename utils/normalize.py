# utils/normalize.py
import re

def normalize_number(s):
    """
    Convert common numeric string forms to float. Returns None if unparsable.
    Examples handled:
      - "₹ 1,23,456" -> 123456.0
      - "(1,234)" -> -1234.0
      - "—", "-", "N/A" -> None
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    if s in {"-", "—", "N/A", "NA", "nil"}:
        return None
    # remove currency symbols and spaces
    s2 = re.sub(r"[^\d\-\.\(\)\/%]", "", s)
    # If value is like (1234) treat as negative
    try:
        if "(" in s2 and ")" in s2:
            s3 = s2.replace("(", "").replace(")", "")
            return float(s3)
        # percent handling (keeps as float percent)
        if s2.endswith("%"):
            val = float(s2.replace("%", ""))
            return val / 100.0
        # plain float
        return float(s2)
    except Exception:
        # try removing commas and other artifacts
        s3 = re.sub(r"[^\d\.\-]", "", s)
        try:
            return float(s3)
        except Exception:
            return None

