"""Parse signed engineering values without discarding exponents or minus signs."""
import ast
import math
import re


def parse_supervisor_response(text):
    def field(label):
        match = re.search(r"^\s*" + re.escape(label) + r"\s*:\s*(.+)$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing supervisor field: {label}")
        return match.group(1).strip()

    score = float(field("Score"))
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Supervisor score must be between zero and one.")
    prompt = field("Improved Prompt")
    properties = []
    for label, size in [("Young's modulus", 3), ("Shear modulus", 3), ("Poisson ratio", 6)]:
        values = ast.literal_eval(field(label))
        if not isinstance(values, (list, tuple)) or len(values) != size:
            raise ValueError(f"{label} must contain {size} values.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError(f"{label} must contain finite numbers.")
        properties.extend(float(value) for value in values)
    return score, prompt, properties
