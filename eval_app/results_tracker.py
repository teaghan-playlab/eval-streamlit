import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional


# The full conversation transcript is never written to CSV. A single transcript
# can run to hundreds of thousands of characters with thousands of embedded line
# breaks, which exceeds spreadsheet cell limits (Excel caps a cell at 32,767
# characters, Google Sheets at 50,000) and corrupts the layout so that only the
# first handful of rows appear to contain data.
EXCLUDED_CSV_COLUMNS = frozenset({"conversation"})

# Cap every exported cell comfortably below Excel's per-cell limit of 32,767
# characters so no single value can overflow into neighbouring cells or rows.
MAX_CSV_CELL_CHARS = 32000

# Every character that a spreadsheet importer might treat as the end of a row.
# Replacing these with spaces keeps each record on a single line.
_LINE_BREAK_CHARS = ("\r\n", "\r", "\n", " ", " ", "\v", "\f", "\t")


def _stringify(value: Any) -> str:
    """Convert any result value to a string for CSV output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _sanitize_cell(value: Any, max_chars: int = MAX_CSV_CELL_CHARS) -> str:
    """Make a single value safe to write into a spreadsheet cell.

    Removes line breaks and other characters that would split a record across
    rows, and truncates over-long text so it cannot exceed spreadsheet cell
    limits.
    """
    text = _stringify(value)
    for line_break in _LINE_BREAK_CHARS:
        text = text.replace(line_break, " ")
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"  # trailing ellipsis marks truncation
    return text


def _export_fieldnames(fieldnames: List[str]) -> List[str]:
    """Drop columns that must never appear in a CSV export."""
    return [name for name in fieldnames if name not in EXCLUDED_CSV_COLUMNS]


def write_results_to_csv(
    results: List[Dict[str, Any]],
    output_path: Path,
    fieldnames: Optional[List[str]] = None,
) -> None:
    """
    Write evaluation results to a CSV file.

    Args:
        results: List of result dictionaries to write
        output_path: Path to the output CSV file
        fieldnames: Optional list of field names to use as CSV headers.
                   If not provided, will use all keys from the first result.
    """
    if not results:
        logging.warning("No results to write to CSV")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine fieldnames
    if fieldnames is None:
        # Get all unique keys from all results
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        fieldnames = sorted(list(all_keys))

    # Drop excluded columns (e.g. the full conversation transcript)
    fieldnames = _export_fieldnames(fieldnames)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for result in results:
            # Sanitize every cell so no value can break the spreadsheet layout
            row = {key: _sanitize_cell(result.get(key, "")) for key in fieldnames}
            writer.writerow(row)

    logging.info(f"Wrote {len(results)} result(s) to {output_path}")


def append_results_to_csv(
    results: List[Dict[str, Any]],
    output_path: Path,
    fieldnames: Optional[List[str]] = None,
) -> None:
    """
    Append evaluation results to an existing CSV file, or create it if it doesn't exist.

    Args:
        results: List of result dictionaries to append
        output_path: Path to the CSV file
        fieldnames: Optional list of field names. If file exists, will use existing headers.
                   If file doesn't exist, will use provided fieldnames or derive from results.
    """
    if not results:
        logging.warning("No results to append to CSV")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    # Determine fieldnames
    if file_exists:
        # Read existing headers so appended rows line up with the current file
        with open(output_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = list(reader.fieldnames or [])
    else:
        # Get all unique keys from all results
        if fieldnames is None:
            all_keys = set()
            for result in results:
                all_keys.update(result.keys())
            fieldnames = sorted(list(all_keys))
        # Drop excluded columns only when creating the file; an existing file's
        # header is respected as-is to keep columns aligned.
        fieldnames = _export_fieldnames(fieldnames)

    # Append to CSV
    with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")

        if not file_exists:
            writer.writeheader()

        for result in results:
            # Sanitize every cell so no value can break the spreadsheet layout
            row = {key: _sanitize_cell(result.get(key, "")) for key in fieldnames}
            writer.writerow(row)

    logging.info(f"Appended {len(results)} result(s) to {output_path}")
