"""Helpers for running evaluations over collections of conversations."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from .evaluation import Evaluator, EvaluationResult


ProgressCallback = Callable[[int, int, Dict[str, Any], Dict[str, Any]], None]
CancelCallback = Callable[[], bool]


def _evaluate_single_conversation(
    evaluator: Evaluator,
    conv: Dict[str, Any],
    idx: int,
    total: int,
    conversation_field: str,
    use_prompt_caching: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate a single conversation and return (original_conv, result_row)."""
    conversation_id = conv.get("conversationId", f"unknown_{idx}")
    logging.info("Evaluating conversation %s/%s: %s", idx, total, conversation_id)

    # Get the conversation text to evaluate
    conversation_text = conv.get(conversation_field, "")

    if not conversation_text:
        logging.warning(
            "Conversation %s has no '%s' field, skipping",
            conversation_id,
            conversation_field,
        )
        # Still add it to results with evaluation failure
        result = conv.copy()
        result.update(
            {
                "evaluatorModel": evaluator.model_name,
                "evaluation_config_file": evaluator.config_file,
                "evaluation_decode_failed": True,
                "evaluation_error": "Missing conversation text",
            }
        )
        return conv, result

    try:
        # Evaluate the conversation
        eval_result: EvaluationResult = evaluator.evaluate(
            conversation_text,
            use_prompt_caching=use_prompt_caching,
        )

        # Combine original conversation data with evaluation results
        result = conv.copy()
        result.update(
            {
                "evaluatorModel": eval_result.model,
                "evaluation_config_file": eval_result.config_file,
                "evaluation_decode_failed": eval_result.decode_failed,
            }
        )

        # Add evaluation response fields if available
        if eval_result.full_response:
            # Add each field from the evaluation response as a separate column
            for field_name, field_value in eval_result.full_response.items():
                result[f"evaluation_{field_name}"] = field_value
        else:
            logging.warning(
                "Conversation %s evaluation returned no response",
                conversation_id,
            )

        return conv, result

    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Error evaluating conversation %s: %s", conversation_id, exc)
        # Add conversation with error information
        result = conv.copy()
        result.update(
            {
                "evaluatorModel": evaluator.model_name,
                "evaluation_config_file": evaluator.config_file,
                "evaluation_decode_failed": True,
                "evaluation_error": str(exc),
            }
        )
        return conv, result


# Signature kept backwards-compatible: new arguments are optional and have defaults.
def evaluate_conversations(
    evaluator: Evaluator,
    conversations: List[Dict[str, Any]],
    conversation_field: str = "conversation",
    progress_callback: Optional[ProgressCallback] = None,
    should_cancel: Optional[CancelCallback] = None,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Evaluate a list of conversations.

    Args:
        evaluator: Initialized Evaluator instance.
        conversations: List of conversation dictionaries (from data_loader).
        conversation_field: Field name containing the conversation text to
            evaluate.
        progress_callback: Optional callable invoked after each conversation
            is processed. Receives:
              (current_index, total_conversations, original_conversation, result_dict).
        should_cancel: Optional callable that returns True to stop evaluation
            early. Checked between conversations (or between completed tasks
            when using multiple workers).
        max_workers: Optional maximum number of worker threads to use for
            parallel evaluation. If None or < 2, evaluation is sequential.

    Returns:
        List of result dictionaries combining conversation data with
        evaluation results.
    """

    results: List[Dict[str, Any]] = []
    total = len(conversations)

    if total == 0:
        return results

    use_prompt_caching = total > 2

    # Normalize worker count
    if max_workers is not None and max_workers < 2:
        max_workers = None

    # Simple sequential path (default)
    if max_workers is None:
        for idx, conv in enumerate(conversations, 1):
            if should_cancel is not None and should_cancel():
                break

            original_conv, result = _evaluate_single_conversation(
                evaluator,
                conv,
                idx,
                total,
                conversation_field,
                use_prompt_caching,
            )
            results.append(result)

            if progress_callback is not None:
                try:
                    progress_callback(idx, total, original_conv, result)
                except Exception:  # pragma: no cover - defensive safeguard
                    # Progress updates should never break evaluation logic
                    logging.exception(
                        "Progress callback failed for conversation %s",
                        original_conv.get("conversationId", f"unknown_{idx}"),
                    )

        return results

    # Parallel path using a thread pool for I/O-bound API calls.
    # Note: order of results is the order in which tasks complete, not input order.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index: Dict[Any, int] = {}
        future_to_conv: Dict[Any, Dict[str, Any]] = {}

        for idx, conv in enumerate(conversations, 1):
            if should_cancel is not None and should_cancel():
                break

            future = executor.submit(
                _evaluate_single_conversation,
                evaluator,
                conv,
                idx,
                total,
                conversation_field,
                use_prompt_caching,
            )
            future_to_index[future] = idx
            future_to_conv[future] = conv

        completed_count = 0

        for future in as_completed(list(future_to_index.keys())):
            if should_cancel is not None and should_cancel():
                break

            idx = future_to_index[future]
            conv = future_to_conv[future]

            try:
                original_conv, result = future.result()
            except Exception as exc:  # pragma: no cover - defensive safeguard
                logging.error(
                    "Worker task failed for conversation %s: %s", idx, exc
                )
                # Fallback: record a generic failure row
                original_conv = conv
                result = conv.copy()
                result.update(
                    {
                        "evaluatorModel": evaluator.model_name,
                        "evaluation_config_file": evaluator.config_file,
                        "evaluation_decode_failed": True,
                        "evaluation_error": str(exc),
                    }
                )

            results.append(result)
            completed_count += 1

            if progress_callback is not None:
                try:
                    progress_callback(completed_count, total, original_conv, result)
                except Exception:  # pragma: no cover - defensive safeguard
                    logging.exception(
                        "Progress callback failed for conversation %s",
                        original_conv.get("conversationId", f"unknown_{idx}"),
                    )

    return results
