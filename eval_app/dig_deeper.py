import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

logger = logging.getLogger(__name__)


def _log_large_block(title: str, text: str) -> None:
    """
    Log a potentially large block of text in a way that's easy to find/copy.

    By default this logs the full text. If `DIG_DEEPER_LOG_MAX_CHARS` is set,
    it truncates to that many characters.
    """
    try:
        max_chars_env = os.getenv("DIG_DEEPER_LOG_MAX_CHARS")
        max_chars = int(max_chars_env) if max_chars_env else None
    except Exception:
        max_chars = None

    if max_chars is not None and max_chars >= 0 and len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[... truncated to {max_chars} chars ...]"

    logger.info("=" * 60)
    logger.info("%s", title)
    logger.info("=" * 60)
    # One giant log line can get truncated by some handlers; chunk defensively.
    chunk_size = 4000
    for i in range(0, len(text), chunk_size):
        logger.info("%s", text[i : i + chunk_size])
    logger.info("=" * 60)


@dataclass
class DigDeeperResult:
    """
    Result of a dig-deeper run.
    - model: model name actually used by the provider
    - config_file: path/name of the config used
    - text: analysis output text
    """

    model: str
    config_file: str
    text: str


@dataclass
class DigDeeperConfig:
    """Configuration for DigDeeper."""

    system_prompt: str
    model_name: str
    config_file: Optional[str] = None


class DigDeeper:
    """
    DigDeeper sends a curated slice of evaluation rows + a user task to Anthropic
    and returns an evidence-based analysis.
    """

    def __init__(
        self,
        config: DigDeeperConfig,
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

        self.system_prompt = (config.system_prompt or "").strip()
        if not self.system_prompt:
            raise ValueError("DigDeeperConfig.system_prompt must be provided")
        self.model_name = config.model_name
        self.config_file = config.config_file or ""
        self.temperature = 0.0

        logging.info("Initialized DigDeeper with model %s", self.model_name)

    def run(
        self,
        *,
        task: str,
        selected_categories: List[str],
        category_definitions: Dict[str, str],
        category_stats: Dict[str, Dict[str, Any]],
        category_filters: Dict[str, str],
        rows: List[Dict[str, Any]],
        include_conversation: bool,
        include_reasoning: bool,
        max_tokens: int = 1500,
        max_conversation_chars: Optional[int] = None,
    ) -> DigDeeperResult:
        """Run a dig-deeper analysis on a curated set of rows."""
        user_payload = self._build_user_payload(
            task=task,
            selected_categories=selected_categories,
            category_definitions=category_definitions,
            category_stats=category_stats,
            category_filters=category_filters,
            rows=rows,
            include_conversation=include_conversation,
            include_reasoning=include_reasoning,
            max_conversation_chars=max_conversation_chars,
        )

        # Log the full system prompt and user payload so callers can verify
        # exactly what was sent to the API. Truncation can be controlled with
        # DIG_DEEPER_LOG_MAX_CHARS if needed.
        _log_large_block("DIG DEEPER — SYSTEM PROMPT", self.system_prompt)
        _log_large_block("DIG DEEPER — USER MESSAGE", user_payload)

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

        return DigDeeperResult(model=model_used, config_file=self.config_file, text=text_out)

    def _build_user_payload(
        self,
        *,
        task: str,
        selected_categories: List[str],
        category_definitions: Dict[str, str],
        category_stats: Dict[str, Dict[str, Any]],
        category_filters: Dict[str, str],
        rows: List[Dict[str, Any]],
        include_conversation: bool,
        include_reasoning: bool,
        max_conversation_chars: Optional[int],
    ) -> str:
        max_conversation_chars = (
            int(max_conversation_chars)
            if isinstance(max_conversation_chars, int)
            else None
        )

        # Create a lightly-sanitized copy of rows to avoid huge payloads.
        payload_rows: List[Dict[str, Any]] = []
        for row in rows:
            out: Dict[str, Any] = dict(row)
            if not include_conversation and "conversation" in out:
                out.pop("conversation", None)
            if include_conversation and max_conversation_chars is not None:
                conv = out.get("conversation")
                if isinstance(conv, str) and len(conv) > max_conversation_chars:
                    out["conversation"] = (
                        conv[:max_conversation_chars]
                        + f"\n\n[... truncated to {max_conversation_chars} chars ...]"
                    )
            if not include_reasoning:
                for k in list(out.keys()):
                    if isinstance(k, str) and k.startswith("evaluation_") and k.endswith("_reasoning"):
                        out.pop(k, None)
            payload_rows.append(out)

        lines: List[str] = []
        lines.append("## Task")
        lines.append(str(task).strip())
        lines.append("")
        lines.append("## Categories")
        lines.append(f"The below category definitions were used to create the initial set of results you are analyzing.")
        for k in selected_categories:
            desc = str((category_definitions or {}).get(k, "") or "").strip()
            lines.append(f"- **`{k}`**: {desc}")
        lines.append("")
        lines.append("## Success rates (across all results)")
        for k in selected_categories:
            st = (category_stats or {}).get(k, {}) or {}
            lines.append(
                f"- **`{k}`**: true=`{st.get('true_count','')}`, false=`{st.get('false_count','')}`, "
                f"rate=`{st.get('success_rate','')}`"
            )
        lines.append("")
        lines.append("## Selection")
        lines.append(f"The below filters have been applied to the original results to select the subset of conversations and evaluations below.")
        lines.append(f"- **Selected categories**: {', '.join([f'`{c}`' for c in selected_categories])}")
        if category_filters:
            lines.append("- **Per-category filters**:")
            for k in selected_categories:
                f = category_filters.get(k, "both")
                lines.append(f"  - `{k}`: `{f}`")
        lines.append(f"- **Include reasoning**: `{bool(include_reasoning)}`")
        lines.append(f"- **Include conversation**: `{bool(include_conversation)}`")
        if include_conversation and max_conversation_chars is not None:
            lines.append(f"- **Conversation truncation**: `{max_conversation_chars}` chars")
        lines.append("")
        lines.append("## Conversations and Evaluations (JSON)")
        lines.append(
            json.dumps(
                payload_rows,
                ensure_ascii=False,
                indent=2,
            )
        )
        return "\n".join(lines)


__all__ = ["DigDeeper", "DigDeeperConfig", "DigDeeperResult"]


