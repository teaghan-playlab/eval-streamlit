#!/usr/bin/env python3
"""
Summarize evaluation results for a single category using Anthropic.

Inputs:
- Config JSON (for categories + summarizer_system_prompt)
- Results JSON (list of dicts, same shape as Streamlit keeps in memory)
- Category key
- User-provided "task" describing what kind of summary to produce
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from eval_app import Summarizer, SummarizerConfig


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _read_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_results_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError("Results JSON must be a list of dicts")


def _compute_success_rate(rows: List[Dict[str, Any]], eval_key: str) -> Tuple[int, int, float]:
    true_count = 0
    false_count = 0
    for row in rows:
        b = _parse_bool(row.get(eval_key))
        if b is True:
            true_count += 1
        elif b is False:
            false_count += 1
    denom = true_count + false_count
    rate = (true_count / float(denom)) if denom > 0 else 0.0
    return true_count, false_count, rate


def _select_examples(
    rows: List[Dict[str, Any]],
    eval_key: str,
    max_examples: int,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    # Keep only rows where the boolean is present.
    usable = [r for r in rows if _parse_bool(r.get(eval_key)) is not None]
    if not usable:
        return []

    if seed is not None:
        random.seed(seed)

    # Try to balance True/False if possible
    trues = [r for r in usable if _parse_bool(r.get(eval_key)) is True]
    falses = [r for r in usable if _parse_bool(r.get(eval_key)) is False]

    if max_examples <= 0:
        return []

    half = max_examples // 2
    picked: List[Dict[str, Any]] = []

    if trues and falses and max_examples >= 2:
        picked.extend(random.sample(trues, min(half, len(trues))))
        picked.extend(random.sample(falses, min(max_examples - len(picked), len(falses))))
    else:
        picked.extend(random.sample(usable, min(max_examples, len(usable))))

    # Stable-ish order for readability
    def _key(r: Dict[str, Any]) -> str:
        return (r.get("createdAt") or "") + (r.get("conversationId") or "")

    return sorted(picked, key=_key)


def _build_user_payload(
    *,
    task: str,
    category_key: str,
    category_description: str,
    true_count: int,
    false_count: int,
    success_rate: float,
    examples: List[Dict[str, str]],
) -> str:
    eval_bool_key = f"evaluation_{category_key}"
    eval_reason_key = f"evaluation_{category_key}_reasoning"

    lines: List[str] = []
    lines.append("## Task")
    lines.append(task.strip())
    lines.append("")
    lines.append("## Category")
    lines.append(f"- **Key**: `{category_key}`")
    lines.append(f"- **Description**: {category_description.strip()}")
    lines.append("")
    lines.append("## Success rate")
    lines.append(f"- **True**: {true_count}")
    lines.append(f"- **False**: {false_count}")
    lines.append(f"- **Rate**: {success_rate:.3f}")
    lines.append("")
    lines.append("## Example rows")
    if not examples:
        lines.append("_No example rows with a boolean value were found for this category._")
        return "\n".join(lines)

    for i, row in enumerate(examples, start=1):
        lines.append(f"### Example {i}")
        lines.append(f"- **conversationId**: `{row.get('conversationId','')}`")
        lines.append(f"- **createdAt**: `{row.get('createdAt','')}`")
        lines.append(f"- **userMessageCount**: `{row.get('userMessageCount','')}`")
        lines.append(f"- **providerMessageCount**: `{row.get('providerMessageCount','')}`")
        lines.append(f"- **avgUserMessageCharacters**: `{row.get('avgUserMessageCharacters','')}`")
        lines.append(f"- **avgProviderMessageCharacters**: `{row.get('avgProviderMessageCharacters','')}`")
        lines.append(f"- **value**: `{row.get(eval_bool_key,'')}`")
        reasoning = row.get(eval_reason_key, "") or ""
        lines.append(f"- **reasoning**: {reasoning.strip()}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a single evaluation category using Anthropic")
    parser.add_argument("config", type=Path, help="Path to evaluator config JSON (includes categories)")
    parser.add_argument("results_json", type=Path, help="Path to results JSON (list of dicts)")
    parser.add_argument("--category", required=True, help="Category key (e.g. gesi_alignment)")
    parser.add_argument("--task", required=True, help="What you want the summarizer to produce")
    parser.add_argument("--max-examples", type=int, default=25, help="Max example rows to include (default: 25)")
    parser.add_argument("--model", default=None, help="Anthropic model (defaults to config.model_name)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for example selection (default: 0)")
    parser.add_argument("--max-tokens", type=int, default=1500, help="Max tokens for summarizer response (default: 1500)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = _read_config(args.config)
    categories: Dict[str, Dict[str, Any]] = config.get("categories", {}) or {}
    if args.category not in categories:
        available = ", ".join(sorted(categories.keys()))
        logging.error("Unknown category '%s'. Available: %s", args.category, available)
        return 2

    category_description = str(categories[args.category].get("description", "") or "")
    system_prompt = str(
        config.get("summarizer_system_prompt")
        or "You are an evaluation summarizer. Produce a concise, evidence-based summary."
    )
    model = args.model or str(config.get("model_name", "") or "")
    if not model:
        logging.error("No model specified and config.model_name is empty.")
        return 2

    rows = _read_results_json(args.results_json)
    eval_bool_key = f"evaluation_{args.category}"

    true_count, false_count, success_rate = _compute_success_rate(rows, eval_bool_key)
    examples = _select_examples(rows, eval_bool_key, args.max_examples, seed=args.seed)

    summarizer = Summarizer(
        SummarizerConfig(
            system_prompt=system_prompt,
            model_name=model,
            config_file=str(args.config),
        )
    )
    result = summarizer.summarize_category(
        task=args.task,
        category_key=args.category,
        category_description=category_description,
        true_count=true_count,
        false_count=false_count,
        success_rate=success_rate,
        examples=examples,
        max_tokens=args.max_tokens,
    )

    print(result.text)
    return 0


if __name__ == "__main__":
    load_dotenv()
    raise SystemExit(main())

