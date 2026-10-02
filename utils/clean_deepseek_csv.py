#!/usr/bin/env python3
"""
Clean DeepSeek CSV: keep only the ~~~...~~~ fenced block in the output column.

Usage (default paths work for this repo):
  python utils/clean_deepseek_csv.py \
      --input results/results_deepseek-reasoner.csv \
      --output results/results_deepseek-reasoner.cleaned.csv

Options:
  --column can be a column name (e.g., output) or a 0-based index (e.g., 1).
  If omitted, the script tries a case-insensitive match for 'output'; otherwise
  it falls back to column index 1 (second column).
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Optional, Union

import pandas as pd


FENCE_PATTERN = re.compile(r"~~~\s*.*?\s*~~~", flags=re.DOTALL)


def extract_fenced_block(text: Optional[str]) -> str:
    """Return the first ~~~...~~~ fenced block (including fences); empty if none.

    The match is DOTALL, so the block can span multiple lines.
    """
    if not isinstance(text, str):
        return ""
    m = FENCE_PATTERN.search(text)
    return m.group(0).strip() if m else ""


def resolve_output_column(df: pd.DataFrame, column: Optional[str]) -> Union[str, int]:
    """Determine which column to clean.

    Priority:
      1) If --column provided: use name or index (int-parsed)
      2) Case-insensitive name match for 'output' in df.columns
      3) Fallback to second column (index 1) if it exists
    """
    if column is not None:
        # Try to parse as integer index
        try:
            idx = int(column)
            if idx < 0 or idx >= len(df.columns):
                raise IndexError(f"Column index {idx} is out of range [0, {len(df.columns)-1}].")
            return idx
        except ValueError:
            # Use as string name
            if column in df.columns:
                return column
            # case-insensitive match attempt
            lower_map = {str(c).lower(): c for c in df.columns}
            if column.lower() in lower_map:
                return lower_map[column.lower()]
            raise KeyError(f"Column '{column}' not found in CSV columns: {list(df.columns)}")

    # Auto-detect 'output' column by name (case-insensitive)
    lower_map = {str(c).lower(): c for c in df.columns}
    if 'output' in lower_map:
        return lower_map['output']

    # Fallback to second column (index 1)
    if len(df.columns) >= 2:
        return 1

    # If there's only one column, use it
    if len(df.columns) == 1:
        return df.columns[0]

    raise RuntimeError("Could not determine an output column to clean.")


def clean_csv(input_path: str, output_path: str, column: Optional[str] = None, drop_missing: bool = False) -> None:
    df = pd.read_csv(input_path)

    out_col = resolve_output_column(df, column)

    # Apply extraction row-wise
    df[out_col] = df[out_col].apply(extract_fenced_block)

    if drop_missing:
        before = len(df)
        df = df[df[out_col].astype(str).str.len() > 0]
        after = len(df)
        print(f"Dropped {before - after} rows without fenced blocks.")

    df.to_csv(output_path, index=False)
    print(f"Cleaned CSV saved to: {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Keep only ~~~ fenced blocks in output column of a CSV.")
    parser.add_argument("--input", "-i", default="results/results_deepseek-reasoner.csv", help="Input CSV path")
    parser.add_argument("--output", "-o", default="results/results_deepseek-reasoner.cleaned.csv", help="Output CSV path")
    parser.add_argument("--column", "-c", default=None, help="Output column name or 0-based index (default: autodetect)")
    parser.add_argument("--drop-missing", action="store_true", help="Drop rows without any fenced block")
    args = parser.parse_args(argv)

    try:
        clean_csv(args.input, args.output, args.column, args.drop_missing)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
