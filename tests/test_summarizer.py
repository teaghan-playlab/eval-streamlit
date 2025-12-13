#!/usr/bin/env python3
"""
Debug/test script for eval_app/summarization.py

Runs the summarizer against a small hardcoded set of result rows.
Loads ANTHROPIC_API_KEY from .env file.
"""

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path so we can import eval_app
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval_app import Summarizer, SummarizerConfig


def main() -> int:
    load_dotenv()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config_path = Path("configs/default_config.json")
    if not config_path.exists():
        logging.error("Config file not found: %s", config_path)
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    system_prompt = cfg.get("summarizer_system_prompt", "")
    model_name = cfg.get("model_name", "")
    if not system_prompt or not model_name:
        logging.error("Config must include summarizer_system_prompt and model_name")
        return 1

    summarizer = Summarizer(
        SummarizerConfig(
            system_prompt=system_prompt,
            model_name=model_name,
            config_file=str(config_path),
        )
    )

    category_key = "gesi_alignment"
    category_description = cfg.get("categories", {}).get(category_key, {}).get("description", "")

    # Hardcoded tiny set of rows in the same shape as Streamlit results
    rows = [
        {
            "conversationId": "conv_1",
            "createdAt": "2025-01-01T00:00:00Z",
            "userMessageCount": 2,
            "providerMessageCount": 2,
            "avgUserMessageCharacters": 120,
            "avgProviderMessageCharacters": 420,
            f"evaluation_{category_key}": True,
            f"evaluation_{category_key}_reasoning": "Includes gender-balanced examples and accessibility considerations.",
        },
        {
            "conversationId": "conv_2",
            "createdAt": "2025-01-02T00:00:00Z",
            "userMessageCount": 1,
            "providerMessageCount": 1,
            "avgUserMessageCharacters": 80,
            "avgProviderMessageCharacters": 260,
            f"evaluation_{category_key}": False,
            f"evaluation_{category_key}_reasoning": "No explicit inclusion strategies; examples are not gender-balanced.",
        },
    ]

    true_count = 1
    false_count = 1
    success_rate = 0.5

    result = summarizer.summarize_category(
        task="Summarize the most common GESI success and failure patterns and give 3 recommendations.",
        category_key=category_key,
        category_description=category_description,
        true_count=true_count,
        false_count=false_count,
        success_rate=success_rate,
        examples=rows,
        max_tokens=800,
    )

    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

