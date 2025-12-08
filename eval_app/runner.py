"""Helpers for running evaluations over collections of conversations."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .evaluation import Evaluator, EvaluationResult


# Signature kept backwards-compatible: the new progress_callback argument is optional.
def evaluate_conversations(
    evaluator: Evaluator,
    conversations: List[Dict[str, Any]],
    conversation_field: str = "conversation",
    progress_callback: Optional[
        Callable[[int, int, Dict[str, Any], Dict[str, Any]], None]
    ] = None,
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

    Returns:
        List of result dictionaries combining conversation data with
        evaluation results.
    """

    results: List[Dict[str, Any]] = []
    total = len(conversations)

    use_prompt_caching = len(conversations) > 2
    for idx, conv in enumerate(conversations, 1):
        conversation_id = conv.get("conversationId", f"unknown_{idx}")
        logging.info(
            "Evaluating conversation %s/%s: %s",
            idx,
            total,
            conversation_id,
        )

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
            results.append(result)

            if progress_callback is not None:
                progress_callback(idx, total, conv, result)

            continue

        try:
            # Evaluate the conversation
            eval_result: EvaluationResult = evaluator.evaluate(conversation_text, use_prompt_caching=use_prompt_caching)

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

            results.append(result)

        except Exception as exc:  # pragma: no cover - defensive logging
            logging.error(
                "Error evaluating conversation %s: %s", conversation_id, exc
            )
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
            results.append(result)

        # Update progress after each processed conversation
        if progress_callback is not None:
            try:
                progress_callback(idx, total, conv, result)
            except Exception:  # pragma: no cover - defensive safeguard
                # Progress updates should never break evaluation logic
                logging.exception("Progress callback failed for conversation %s", conversation_id)

    return results
