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
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from evaluator import Evaluator, EvaluatorConfig, EvaluationResult
from data_loader import load_conversations_from_file, load_all_conversations, get_all_headers, order_csv_headers
from results_tracker import write_results_to_csv, append_results_to_csv

from dotenv import load_dotenv


def load_config(config_path: Path) -> EvaluatorConfig:
    """
    Load evaluator configuration from JSON file.

    Args:
        config_path: Path to the config JSON file

    Returns:
        EvaluatorConfig object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    config = EvaluatorConfig(**config_data)
    # Set config_file path for tracking
    config.config_file = str(config_path)
    return config


def evaluate_conversations(
    evaluator: Evaluator,
    conversations: List[Dict[str, Any]],
    conversation_field: str = "conversation",
) -> List[Dict[str, Any]]:
    """
    Evaluate a list of conversations.

    Args:
        evaluator: Initialized Evaluator instance
        conversations: List of conversation dictionaries from data_loader
        conversation_field: Field name containing the conversation text to evaluate

    Returns:
        List of result dictionaries combining conversation data with evaluation results
    """
    results = []

    for idx, conv in enumerate(conversations, 1):
        conversation_id = conv.get("conversationId", f"unknown_{idx}")
        logging.info(f"Evaluating conversation {idx}/{len(conversations)}: {conversation_id}")

        # Get the conversation text to evaluate
        conversation_text = conv.get(conversation_field, "")

        if not conversation_text:
            logging.warning(
                f"Conversation {conversation_id} has no '{conversation_field}' field, skipping"
            )
            # Still add it to results with evaluation failure
            result = conv.copy()
            result.update({
                "evaluatorModel": evaluator.model_name,
                "evaluation_config_file": evaluator.config_file,
                "evaluation_decode_failed": True,
                "evaluation_error": "Missing conversation text",
            })
            results.append(result)
            continue

        try:
            # Evaluate the conversation
            eval_result: EvaluationResult = evaluator.evaluate(conversation_text)

            # Combine original conversation data with evaluation results
            result = conv.copy()
            result.update({
                "evaluatorModel": eval_result.model,
                "evaluation_config_file": eval_result.config_file,
                "evaluation_decode_failed": eval_result.decode_failed,
            })

            # Add evaluation response fields if available
            if eval_result.full_response:
                # Add each field from the evaluation response as a separate column
                for field_name, field_value in eval_result.full_response.items():
                    result[f"evaluation_{field_name}"] = field_value
            else:
                logging.warning(f"Conversation {conversation_id} evaluation returned no response")

            results.append(result)

        except Exception as e:
            logging.error(f"Error evaluating conversation {conversation_id}: {e}")
            # Add conversation with error information
            result = conv.copy()
            result.update({
                "evaluatorModel": evaluator.model_name,
                "evaluation_config_file": evaluator.config_file,
                "evaluation_decode_failed": True,
                "evaluation_error": str(e),
            })
            results.append(result)

    return results


def main():
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
