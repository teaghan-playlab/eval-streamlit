import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from anthropic import Anthropic


@dataclass
class SummaryResult:
    """
    Result of a summarization run.
    - model: model name actually used by the provider
    - config_file: path/name of the config used
    - text: summarizer output text
    """

    model: str
    config_file: str
    text: str


@dataclass
class SummarizerConfig:
    """Configuration for the Summarizer."""

    system_prompt: str
    model_name: str
    config_file: Optional[str] = None


class Summarizer:
    """
    Summarizer that sends category-level evaluation results to Anthropic Claude API
    and returns an evidence-based summary.
    """

    def __init__(
        self,
        config: SummarizerConfig,
        client: Optional[Anthropic] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY must be set as environment variable "
                    "or passed as api_key parameter to use Anthropic API directly."
                )
            self.client = Anthropic(api_key=api_key)

        if not config.system_prompt:
            raise ValueError("SummarizerConfig.system_prompt must be provided")

        self.system_prompt = config.system_prompt
        self.model_name = config.model_name
        self.config_file = config.config_file or ""
        self.temperature = 0.0

        logging.info("Initialized Summarizer with model %s", self.model_name)

    def summarize_category(
        self,
        *,
        task: str,
        category_key: str,
        category_description: str,
        true_count: int,
        false_count: int,
        success_rate: float,
        examples: List[Dict[str, Any]],
        max_tokens: int = 1500,
    ) -> SummaryResult:
        """Summarize a single category with rate + examples."""
        user_payload = self._build_user_payload(
            task=task,
            category_key=category_key,
            category_description=category_description,
            true_count=true_count,
            false_count=false_count,
            success_rate=success_rate,
            examples=examples,
        )

        logging.info("Summarizer system prompt preview:\n%s", self.system_prompt[:2000])
        logging.info("Summarizer payload preview:\n%s", user_payload[:2000])

        resp = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_payload}],
        )

        out_parts: List[str] = []
        for block in getattr(resp, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                out_parts.append(text)

        text_out = "\n".join(out_parts).strip()
        model_used = getattr(resp, "model", self.model_name)

        return SummaryResult(model=model_used, config_file=self.config_file, text=text_out)

    def _build_user_payload(
        self,
        *,
        task: str,
        category_key: str,
        category_description: str,
        true_count: int,
        false_count: int,
        success_rate: float,
        examples: List[Dict[str, Any]],
    ) -> str:
        eval_bool_key = f"evaluation_{category_key}"
        eval_reason_key = f"evaluation_{category_key}_reasoning"

        lines: List[str] = []
        lines.append("## Task")
        lines.append(str(task).strip())
        lines.append("")
        lines.append("## Category")
        lines.append(f"- **Key**: `{category_key}`")
        lines.append(f"- **Description**: {str(category_description).strip()}")
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
            lines.append(f"- **reasoning**: {str(reasoning).strip()}")
            lines.append("")

        return "\n".join(lines)


__all__ = ["Summarizer", "SummarizerConfig", "SummaryResult"]

