#!/usr/bin/env python3
"""
Main script to run evaluations on conversation data.

Handles:
- Loading configuration
- Loading conversations from JSON files
- Running evaluations on each conversation
- Tracking and saving results to CSV
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from eval_app import (
    Evaluator,
    load_config,
    load_conversations_from_file,
    load_all_conversations,
    get_all_headers,
    order_csv_headers,
    write_results_to_csv,
    append_results_to_csv,
    evaluate_conversations,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run evaluations on conversation data from JSON files"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the evaluator config JSON file",
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to JSON file or directory containing JSON files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results.csv"),
        help="Output CSV file path (default: results.csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV file instead of overwriting",
    )
    parser.add_argument(
        "--conversation-field",
        default="conversation",
        help="Field name containing conversation text to evaluate (default: 'conversation')",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of conversations to evaluate (default: evaluate all)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        # Load configuration
        logging.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Initialize evaluator
        logging.info("Initializing evaluator")
        evaluator = Evaluator(config)

        # Load conversations
        data_path = Path(args.data_path)
        if data_path.is_file():
            logging.info(f"Loading conversations from file: {data_path}")
            conversations = load_conversations_from_file(data_path)
        elif data_path.is_dir():
            logging.info(f"Loading conversations from directory: {data_path}")
            conversations = load_all_conversations(data_path)
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")

        if not conversations:
            logging.error("No conversations found to evaluate")
            return 1

        # Apply max-samples limit if specified
        original_count = len(conversations)
        if args.max_samples is not None and args.max_samples > 0:
            conversations = conversations[:args.max_samples]
            logging.info(
                f"Limiting evaluation to {len(conversations)} conversation(s) "
                f"(requested: {args.max_samples}, available: {original_count})"
            )
        else:
            logging.info(f"Found {len(conversations)} conversation(s) to evaluate")

        # Run evaluations
        results = evaluate_conversations(
            evaluator,
            conversations,
            conversation_field=args.conversation_field,
        )

        # Get all headers for CSV and order them
        all_headers = get_all_headers(results)
        headers = order_csv_headers(all_headers)

        # Write results to CSV
        if args.append:
            logging.info(f"Appending results to {args.output}")
            append_results_to_csv(results, args.output, fieldnames=headers)
        else:
            logging.info(f"Writing results to {args.output}")
            write_results_to_csv(results, args.output, fieldnames=headers)

        logging.info(f"Successfully evaluated {len(results)} conversation(s)")
        logging.info(f"Results saved to {args.output}")

        return 0

    except Exception as e:
        logging.exception(f"Error running evaluations: {e}")
        return 1


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
