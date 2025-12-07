import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional


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

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        
        for result in results:
            # Convert any non-string values to strings for CSV compatibility
            row = {}
            for key in fieldnames:
                value = result.get(key, "")
                # Handle None, bool, list, dict
                if value is None:
                    row[key] = ""
                elif isinstance(value, bool):
                    row[key] = str(value).lower()
                elif isinstance(value, (list, dict)):
                    row[key] = str(value)
                else:
                    row[key] = str(value)
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
        # Read existing headers
        with open(output_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
    else:
        # Get all unique keys from all results
        if fieldnames is None:
            all_keys = set()
            for result in results:
                all_keys.update(result.keys())
            fieldnames = sorted(list(all_keys))

    # Append to CSV
    with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        
        if not file_exists:
            writer.writeheader()
        
        for result in results:
            # Convert any non-string values to strings for CSV compatibility
            row = {}
            for key in fieldnames:
                value = result.get(key, "")
                # Handle None, bool, list, dict
                if value is None:
                    row[key] = ""
                elif isinstance(value, bool):
                    row[key] = str(value).lower()
                elif isinstance(value, (list, dict)):
                    row[key] = str(value)
                else:
                    row[key] = str(value)
            writer.writerow(row)

    logging.info(f"Appended {len(results)} result(s) to {output_path}")
