"""Streamlit app for running conversation evaluations."""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
import threading

from dotenv import load_dotenv
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from eval_app import (
    Evaluator,
    EvaluatorConfig,
    Summarizer,
    SummarizerConfig,
    categories_to_fields,
    evaluate_conversations,
    fields_to_categories,
    get_all_headers,
    load_conversations_from_file,
    order_csv_headers,
    slugify_label,
    write_results_to_csv,
)

# Load environment variables from a .env file (project root or current dir)
load_dotenv()

# Set page config must be the first Streamlit command
st.set_page_config(page_title="EvalLab", page_icon="📋", layout="wide")

# User-facing limits (kept in one place for easy adjustment)
MAX_CATEGORIES = int(os.getenv("MAX_CATEGORIES", 10))
MAX_CONVERSATIONS = int(os.getenv("MAX_CONVERSATIONS", 40))


def _project_root() -> Path:
    return Path(__file__).parent


def load_base_config() -> Dict[str, Any]:
    """Load the default config JSON once and cache it."""
    if "base_config" in st.session_state and st.session_state["base_config"] is not None:
        return st.session_state["base_config"]

    config_file_name = os.getenv("DEFAULT_CONFIG_FILE", "default_config.json")
    config_path = _project_root() / "configs" / config_file_name
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    st.session_state["base_config"] = base_config
    # Track the config file name
    if st.session_state.get("config_file_name") is None:
        st.session_state["config_file_name"] = f"configs/{config_file_name}"
    return base_config


def ensure_session_state_keys() -> None:
    """Initialize keys the app relies on."""
    defaults = {
        "access_granted": False,
        "base_config": None,
        "config": None,  # {"system_prompt": str, "model_name": str}
        "config_file_name": None,  # Name/path of the config file being used
        "categories": [],  # [{"label": str, "key": str, "description": str}]
        "evaluator_context": "",  # Additional context for evaluator
        "conversations": [],
        "num_to_evaluate": None,
        "sample_randomly": False,
        "results": [],
        "csv_bytes": None,
        # Result inspection state
        "selected_result_idx": 0,
        # Summarization state
        "summarizer_selected_category_key": None,
        "summarizer_task": "",
        "summarizer_stats": None,
        "summarizer_output": None,
        # Long-running evaluation state
        "is_running": False,
        "cancel_event": None,  # threading.Event for cancellation
        "progress_dict": None,  # Dict for thread-safe progress updates
        "progress": 0.0,
        "status_text": "",
        "detail_text": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_access_gate() -> None:
    """Render the access code input and enforce env-based gate."""
    required_code = os.getenv("EVAL_APP_ACCESS_CODE")

    if not required_code:
        st.error("EVAL_APP_ACCESS_CODE is not set in the environment.")
        st.stop()

    if st.session_state.get("access_granted"):
        return

    st.markdown("### Access")
    code_input = st.text_input("Enter access code", type="password")

    if not code_input:
        st.info("Access code is required to use this app.")
        st.stop()

    if code_input == required_code:
        st.session_state["access_granted"] = True
        st.success("Access granted.")
        st.rerun()
    else:
        st.error("Incorrect access code.")
        st.stop()


def _initialize_config_and_categories_from_base() -> None:
    """Ensure there is a current config and categories, seeded from base config."""
    base_config = load_base_config()

    if st.session_state["config"] is None:
        st.session_state["config"] = {
            "system_prompt": base_config.get("system_prompt", ""),
            "model_name": base_config.get("model_name", ""),
        }

    if not st.session_state["categories"]:
        categories_data = base_config.get("categories", {})
        _set_categories(fields_to_categories(categories_data))

    # Initialize evaluator context from base config if not already set
    if st.session_state.get("evaluator_context") == "":
        context = base_config.get("context", "")
        st.session_state["evaluator_context"] = context

    # Initialize summarizer task from base config if not already set
    if st.session_state.get("summarizer_task") == "":
        st.session_state["summarizer_task"] = str(base_config.get("summarizer_task", "") or "")


def _parse_bool(value: Any) -> bool | None:
    """Parse booleans from either bools or common string values."""
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


def _mean_int(values: List[Any]) -> int | None:
    nums: List[int] = []
    for v in values:
        try:
            nums.append(int(v))
        except Exception:
            continue
    if not nums:
        return None
    return int(round(sum(nums) / float(len(nums))))


def _select_summary_examples(
    *,
    rows_true: List[Dict[str, Any]],
    rows_false: List[Dict[str, Any]],
    max_examples: int,
) -> List[Dict[str, Any]]:
    """Pick a roughly balanced set of examples (True/False) in a stable order."""
    max_examples = max(1, int(max_examples))
    # Keep deterministic ordering: createdAt then conversationId
    def _key(r: Dict[str, Any]) -> str:
        return str(r.get("createdAt") or "") + str(r.get("conversationId") or "")

    rows_true = sorted(rows_true, key=_key)
    rows_false = sorted(rows_false, key=_key)

    if not rows_false:
        return rows_true[:max_examples]
    if not rows_true:
        return rows_false[:max_examples]

    half = max_examples // 2
    picked = rows_true[:half] + rows_false[: (max_examples - half)]
    return picked


def _clear_summarizer_outputs() -> None:
    """Clear cached summarizer outputs when inputs change."""
    st.session_state["summarizer_stats"] = None
    st.session_state["summarizer_output"] = None


def _set_categories(categories: List[Dict[str, str]]) -> None:
    """Set categories and mark that widgets should be re-initialized.

    We don't touch the text-input widgets here. Instead, we set a flag so
    that on the *next* run, the function that renders the text boxes can
    copy labels/descriptions from `categories` into `st.session_state`
    for each `cat_label_*` / `cat_desc_*` key.
    """
    st.session_state["categories"] = categories
    # Signal that on the next render, text boxes should be recreated from
    # the categories data (used by `_render_category_textboxes`).
    st.session_state["categories_reset"] = True


def _render_category_textboxes(
    categories: List[Dict[str, str]]
) -> tuple[List[Dict[str, str]], int | None]:
    """Create text boxes for each category based on session state.

    This is the single place that renders the category text inputs.
    It uses `st.session_state["categories"]` as the source of truth and
    keeps the per-widget keys (`cat_label_*`, `cat_desc_*`) in sync.
    """

    # If categories were just replaced programmatically (upload/reset/add/remove),
    # initialize widget values from the new categories.
    if st.session_state.get("categories_reset"):
        for idx, cat in enumerate(categories):
            st.session_state[f"cat_label_{idx}"] = cat.get("label", "")
            st.session_state[f"cat_desc_{idx}"] = cat.get("description", "")
        st.session_state["categories_reset"] = False

    updated_categories: List[Dict[str, str]] = []
    used_keys: List[str] = []
    remove_idx: int | None = None

    for idx, cat in enumerate(categories):
        col_label, col_remove = st.columns([5, 1])

        with col_label:
            st.markdown(f"**Category {idx + 1}**")

        with col_remove:
            if len(categories) > 1:
                # If this button is clicked, we remember which index to drop.
                if st.button("Remove", key=f"remove_{idx}", use_container_width=True):
                    remove_idx = idx

        label = st.text_input(
            "Category name",
            key=f"cat_label_{idx}",
        )
        key = slugify_label(label, used_keys)
        used_keys.append(key)

        description = st.text_area(
            "Condition for TRUE",
            key=f"cat_desc_{idx}",
        )

        updated_categories.append(
            {
                "label": label,
                "key": key,
                "description": description,
            }
        )

        st.markdown("---")

    return updated_categories, remove_idx

def _on_uploaded_config_change() -> None:
    """Handle a newly uploaded evaluator config file via on_change.

    This reads the uploaded file from session_state and updates both
    `config` and `categories`, then triggers a rerun so the editor
    reflects the new values.
    """
    uploaded = st.session_state.get("uploaded_config_file")
    if uploaded is None:
        return

    try:
        config_data = json.loads(uploaded.getvalue().decode("utf-8"))
    except json.JSONDecodeError:
        st.error("Uploaded config is not valid JSON.")
        return

    system_prompt = str(config_data.get("system_prompt", ""))
    model_name = str(config_data.get("model_name", ""))
    if not system_prompt or not model_name:
        st.error("Config must include 'system_prompt' and 'model_name'.")
        return

    # Update core config
    st.session_state["config"] = {
        "system_prompt": system_prompt,
        "model_name": model_name,
    }

    # Track the uploaded config file name
    uploaded_file_name = uploaded.name if hasattr(uploaded, "name") else "uploaded_config.json"
    st.session_state["config_file_name"] = uploaded_file_name

    # Update categories from uploaded config (or a single default category).
    # We intentionally do NOT call `st.rerun()` here: Streamlit will rerun
    # the script automatically after this on_change callback returns.
    categories_data = config_data.get("categories", {})
    if categories_data:
        _set_categories(fields_to_categories(categories_data))
    else:
        _set_categories(
            [{"label": "Category 1", "key": "category_1", "description": ""}]
        )

    # Update evaluator context from uploaded config
    context = config_data.get("context", "")
    st.session_state["evaluator_context"] = context


def render_categories_editor() -> None:
    """Render the Evaluation Categories editor UI."""
    # Ensure we are initialized from the base config on first load.
    _initialize_config_and_categories_from_base()

    # Optional: allow overriding via uploaded config JSON.
    st.file_uploader(
        "Upload a custom evaluator config JSON",
        type="json",
        key="uploaded_config_file",
        on_change=_on_uploaded_config_change,
    )

    # Simple hint about current model
    cfg = st.session_state["config"] or {}
    #st.caption(f"Current model: `{cfg.get('model_name', '')}`")

    categories: List[Dict[str, str]] = st.session_state["categories"] or []

    if not categories:
        categories = [{"label": "Category 1", "key": "category_1", "description": ""}]

    st.write("Define 1–10 evaluation categories. Each category becomes a boolean field in the evaluator schema.")

    # Reset button
    if st.button("Reset to default config"):
        base = load_base_config()
        categories_data = base.get("categories", {})
        categories = fields_to_categories(categories_data)
        _set_categories(categories)
        # Reset evaluator context to default
        st.session_state["evaluator_context"] = base.get("context", "")
        # Reset config file name to default
        config_file_name = os.getenv("DEFAULT_CONFIG_FILE", "default_config.json")
        st.session_state["config_file_name"] = f"configs/{config_file_name}"
        st.rerun()

    # Render all category text boxes in one dedicated function.
    updated_categories, remove_idx = _render_category_textboxes(categories)

    # Handle removal (structural change -> update categories + rerun)
    if remove_idx is not None:
        new_categories = [
            cat for idx, cat in enumerate(updated_categories) if idx != remove_idx
        ]
        _set_categories(new_categories)
        st.rerun()

    # Add category button below the list
    if len(updated_categories) < MAX_CATEGORIES:
        if st.button("Add category", width="stretch"):
            updated_categories.append(
                {
                    "label": f"Category {len(updated_categories) + 1}",
                    "key": f"category_{len(updated_categories) + 1}",
                    "description": "",
                }
            )
            _set_categories(updated_categories)
            st.rerun()

    # Enforce limits
    if len(updated_categories) < 1:
        st.warning(f"At least 1 category is required.")
    if len(updated_categories) > MAX_CATEGORIES:
        st.warning(f"At most {MAX_CATEGORIES} categories are supported.")

    # Persist latest edits from the text boxes
    st.session_state["categories"] = updated_categories

    # Allow user to download the current evaluator config (system prompt,
    # model name, categories, and context) as JSON for reuse in future runs.
    cfg = st.session_state.get("config") or {}
    config_payload = {
        "system_prompt": cfg.get("system_prompt", ""),
        "model_name": cfg.get("model_name", ""),
        "categories": categories_to_fields(updated_categories),
        "context": st.session_state.get("evaluator_context", ""),
    }
    config_json = json.dumps(config_payload, indent=2)

    st.download_button(
        label="Download current config JSON",
        data=config_json,
        file_name="evaluator_config.json",
        mime="application/json",
    )


def render_conversation_uploader_and_selector() -> None:
    """Render uploader for conversations JSON and selection controls."""
    uploaded = st.file_uploader("Upload conversations JSON file extracted from Playlab", type="json")

    if uploaded is not None:
        # Persist uploaded file to a temporary path so we can reuse the existing loader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        with st.spinner("Loading conversations..."):
            conversations = load_conversations_from_file(tmp_path)

        if not conversations:
            st.error("No conversations found in the uploaded file.")
        else:
            st.session_state["conversations"] = conversations

    conversations: List[Dict[str, Any]] = st.session_state["conversations"]

    if not conversations:
        st.info("Upload a conversations JSON file to begin.")
        return

    st.success(f"Loaded {len(conversations)} conversations.")

    # Show a small preview of the first few conversations (metadata only)
    preview_count = min(5, len(conversations))
    preview_rows: List[Dict[str, Any]] = []
    st.markdown("Conversations preview:")
    for conv in conversations[:preview_count]:
        preview_rows.append(
            {
                "conversationId": conv.get("conversationId", ""),
                "createdAt": conv.get("createdAt", ""),
                "userMessageCount": conv.get("userMessageCount", 0),
                "providerMessageCount": conv.get("providerMessageCount", 0),
            }
        )

    if preview_rows:
        st.dataframe(preview_rows, width="stretch")

    max_n = min(MAX_CONVERSATIONS, len(conversations))
    default_n = min(10, max_n)

    st.session_state["num_to_evaluate"] = st.slider(
        "Number of conversations to evaluate",
        min_value=1,
        max_value=max_n,
        value=default_n,
    )

    st.session_state["sample_randomly"] = st.checkbox(
        f"Randomly sample conversations (instead of taking the first {st.session_state['num_to_evaluate']})",
        value=False,
    )


def validate_ready_to_run() -> tuple[bool, str | None]:
    """Check whether all prerequisites to run evaluation are satisfied.
    
    Returns:
        Tuple of (is_valid, error_message). If is_valid is True, error_message is None.
    """
    logger.debug("Validating readiness to run...")
    
    if not st.session_state.get("access_granted"):
        logger.debug("Validation failed: access not granted")
        return False, "Access code is required. Please enter the access code."

    cfg = st.session_state.get("config") or {}
    if not cfg.get("model_name"):
        logger.debug("Validation failed: model_name missing")
        return False, "Model name is required. Please configure the model name."
    if not cfg.get("system_prompt"):
        logger.debug("Validation failed: system_prompt missing")
        return False, "System prompt is required. Please configure the system prompt."

    categories = st.session_state.get("categories") or []
    if len(categories) < 1:
        logger.debug(f"Validation failed: no categories (count: {len(categories)})")
        return False, f"At least 1 category is required. Please add at least one evaluation category."
    if len(categories) > MAX_CATEGORIES:
        logger.debug(f"Validation failed: too many categories ({len(categories)} > {MAX_CATEGORIES})")
        return False, f"Too many categories. Maximum {MAX_CATEGORIES} categories are supported. Please remove some categories."

    conversations = st.session_state.get("conversations") or []
    if not conversations:
        logger.debug("Validation failed: no conversations loaded")
        return False, "No conversations loaded. Please upload a conversations JSON file."

    num = st.session_state.get("num_to_evaluate")
    if not isinstance(num, int) or num < 1:
        logger.debug(f"Validation failed: invalid num_to_evaluate ({num})")
        return False, "Invalid number of conversations to evaluate. Please select a valid number."
    if num > min(MAX_CONVERSATIONS, len(conversations)):
        logger.debug(f"Validation failed: num ({num}) exceeds max ({min(MAX_CONVERSATIONS, len(conversations))})")
        return False, f"Number to evaluate ({num}) exceeds the maximum allowed ({min(MAX_CONVERSATIONS, len(conversations))}). Please select a smaller number."

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.debug("Validation failed: ANTHROPIC_API_KEY not set")
        return False, "ANTHROPIC_API_KEY is not set in the environment. Please set the API key."

    logger.debug("Validation passed: ready to run")
    return True, None


def _run_evaluations_worker(
    config: Dict[str, Any],
    categories: List[Dict[str, str]],
    conversations: List[Dict[str, Any]],
    num_to_evaluate: int,
    sample_randomly: bool,
    evaluator_context: str,
    config_file_name: str,
    cancel_event: threading.Event,
    progress_dict: Dict[str, Any],
) -> None:
    """Background worker: run evaluations and update progress_dict.

    This function is intended to be executed in a separate thread. It should
    not access st.session_state directly. Instead, it updates progress_dict
    which the main thread will sync to session_state.
    """
    logger.info("Worker thread started")
    logger.info(f"Config: model_name={config.get('model_name')}, system_prompt length={len(config.get('system_prompt', ''))}")
    logger.info(f"Categories count: {len(categories)}")
    logger.info(f"Conversations count: {len(conversations)}")
    logger.info(f"Num to evaluate: {num_to_evaluate}, sample_randomly: {sample_randomly}")
    
    try:
        logger.info("Converting categories to fields...")
        categories_fields = categories_to_fields(categories)
        logger.info(f"Categories fields: {list(categories_fields.keys())}")

        logger.info("Creating EvaluatorConfig...")
        eval_config = EvaluatorConfig(
            system_prompt=config["system_prompt"],
            model_name=config["model_name"],
            categories=categories_fields,
            config_file=config_file_name,
            context=evaluator_context,
        )

        logger.info("Creating Evaluator instance...")
        evaluator = Evaluator(eval_config)
        logger.info("Evaluator created successfully")

        max_n = min(MAX_CONVERSATIONS, len(conversations))
        num = max(1, min(num_to_evaluate, max_n))
        logger.info(f"Will evaluate {num} conversations (max_n={max_n}, requested={num_to_evaluate})")

        if sample_randomly:
            logger.info("Sampling conversations randomly...")
            random.seed()
            selected = random.sample(conversations, num)
        else:
            logger.info("Taking first N conversations...")
            selected = conversations[:num]
        
        logger.info(f"Selected {len(selected)} conversations for evaluation")

        # Reset progress state
        logger.info("Initializing progress_dict...")
        progress_dict["progress"] = 0.0
        progress_dict["status_text"] = "Starting evaluation..."
        progress_dict["detail_text"] = ""
        progress_dict["results"] = []
        progress_dict["csv_bytes"] = None
        logger.info("Progress dict initialized")

        def _should_cancel() -> bool:
            cancelled = cancel_event.is_set()
            if cancelled:
                logger.info("Cancel requested")
            return cancelled

        def _progress_callback(
            idx: int,
            total_count: int,
            original_conv: Dict[str, Any],
            result_row: Dict[str, Any],
        ) -> None:
            """Update numeric progress and last status in progress_dict."""
            if total_count > 0:
                progress_dict["progress"] = idx / float(total_count)

            conversation_id = original_conv.get(
                "conversationId", f"conversation_{idx}"
            )
            progress_dict["status_text"] = (
                f"Evaluated {idx} of {total_count} conversations"
            )

            decode_failed = result_row.get("evaluation_decode_failed", False)
            error_message = result_row.get("evaluation_error")

            if decode_failed or error_message:
                message = error_message or (
                    "Evaluation decode failed for this conversation."
                )
                progress_dict["detail_text"] = (
                    f"Evaluation for `{conversation_id}` failed: {message}"
                )
                logger.warning(f"Evaluation failed for {conversation_id}: {message}")
            else:
                progress_dict["detail_text"] = (
                    f"Evaluation for `{conversation_id}` completed successfully."
                )
            
            logger.info(f"Progress: {idx}/{total_count} ({conversation_id})")

        # Determine worker parallelism from environment (default to 1 = sequential)
        max_workers_env = os.getenv("EVAL_MAX_WORKERS", "1")
        try:
            max_workers = int(max_workers_env)
        except ValueError:
            max_workers = 1

        if max_workers < 1:
            max_workers = 1
        logger.info(f"Using max_workers: {max_workers}")

        logger.info("Starting evaluate_conversations...")
        results = evaluate_conversations(
            evaluator,
            selected,
            conversation_field="conversation",
            progress_callback=_progress_callback,
            should_cancel=_should_cancel,
            max_workers=max_workers,
        )
        logger.info(f"evaluate_conversations completed. Got {len(results) if results else 0} results")

        progress_dict["results"] = results

        if results:
            logger.info("Building CSV from results...")
            # Build CSV only if we have at least one result row
            headers = order_csv_headers(get_all_headers(results))
            logger.info(f"CSV headers: {headers[:5]}... ({len(headers)} total)")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp_path = Path(tmp.name)
            write_results_to_csv(results, tmp_path, fieldnames=headers)
            logger.info(f"CSV written to {tmp_path}")

            with open(tmp_path, "rb") as f:
                progress_dict["csv_bytes"] = f.read()
            logger.info(f"CSV bytes loaded: {len(progress_dict['csv_bytes'])} bytes")

        # Mark completion
        if not cancel_event.is_set():
            progress_dict["progress"] = 1.0
            if not progress_dict.get("status_text"):
                progress_dict["status_text"] = "Evaluation complete."
            logger.info("Evaluation completed successfully")
        else:
            logger.info("Evaluation was cancelled")
    except Exception as e:
        logger.error(f"Error in worker thread: {e}", exc_info=True)
        progress_dict["status_text"] = f"Error: {str(e)}"
        progress_dict["detail_text"] = f"Worker thread encountered an error: {type(e).__name__}"
    finally:
        # Always clear running flag when worker exits
        logger.info("Worker thread finishing, setting is_running=False")
        progress_dict["is_running"] = False


def render_results_section() -> None:
    """Render results preview and CSV download controls."""
    results: List[Dict[str, Any]] = st.session_state.get("results") or []
    csv_bytes = st.session_state.get("csv_bytes")

    if not results:
        st.info("No evaluation results yet. Run evaluations to see results here.")
        return
    st.markdown("**Preview:**")

    # Show a small table of key columns (omit full conversation for readability)
    display_rows: List[Dict[str, Any]] = []
    for row in results[:10]:
        base_row: Dict[str, Any] = {
            "conversationId": row.get("conversationId", ""),
            #"evaluationSuccessful": not row.get("evaluation_decode_failed", False),
        }

        # Include boolean evaluation category results (omit reasoning text fields)
        for key, value in row.items():
            if (
                key.startswith("evaluation_")
                and not key.endswith("_reasoning")
                and not key.endswith("_decode_failed")
                and isinstance(value, bool)
            ):
                base_row[key] = value

        display_rows.append(base_row)

    st.dataframe(display_rows, width="stretch")

    total_results = len(results)
    idx_from_state = st.session_state.get("selected_result_idx", 0)
    try:
        idx_from_state = int(idx_from_state)
    except Exception:
        idx_from_state = 0
    selected_idx = max(0, min(total_results - 1, idx_from_state))
    st.session_state["selected_result_idx"] = selected_idx

    def _format_result_option(i: int) -> str:
        row = results[i] if 0 <= i < total_results else {}
        cid = row.get("conversationId", "") or f"row_{i+1}"
        app_name = row.get("appName", "")
        created_at = row.get("createdAt", "")
        prefix = f"{i+1}/{total_results} — {cid}"
        suffix_parts = [p for p in [app_name, created_at] if p]
        return f"{prefix} ({' | '.join(suffix_parts)})" if suffix_parts else prefix

    @st.dialog("Evaluation details", width="large")
    def _open_result_dialog() -> None:
        # Conversation selection INSIDE the dialog
        picked_idx = st.selectbox(
            "Select a conversation",
            options=list(range(total_results)),
            index=selected_idx,
            format_func=_format_result_option,
        )
        st.session_state["selected_result_idx"] = int(picked_idx)
        row: Dict[str, Any] = results[int(picked_idx)]

        conversation_id = row.get("conversationId", "") or f"row_{int(picked_idx)+1}"
        st.markdown(f"**Conversation:** `{conversation_id}`")

        # Metadata (counts + averages)
        meta = {
            "userMessageCount": row.get("userMessageCount", 0),
            "providerMessageCount": row.get("providerMessageCount", 0),
            "avgUserMessageCharacters": row.get("avgUserMessageCharacters", 0),
            "avgProviderMessageCharacters": row.get("avgProviderMessageCharacters", 0),
        }
        st.markdown(
            f"**User messages:** `{meta['userMessageCount']}`  \n"
            f"**Provider messages:** `{meta['providerMessageCount']}`  \n"
            f"**Avg user characters:** `{meta['avgUserMessageCharacters']}`  \n"
            f"**Avg provider characters:** `{meta['avgProviderMessageCharacters']}`"
        )

        # Evaluation fields: render a markdown summary (boolean + reasoning)
        reasoning_suffix = "_reasoning"
        eval_prefix = "evaluation_"
        reasoning_keys = [
            k for k in row.keys()
            if isinstance(k, str) and k.startswith(eval_prefix) and k.endswith(reasoning_suffix)
        ]
        with st.expander("Evaluation", expanded=False):
            if reasoning_keys:
                lines: List[str] = []
                for rk in sorted(reasoning_keys):
                    base = rk[len(eval_prefix):-len(reasoning_suffix)]
                    bk = f"{eval_prefix}{base}"
                    value = row.get(bk, "")
                    reasoning = row.get(rk, "")
                    lines.append(f"#### {base}")
                    lines.append(f"- **Value**: `{value}`")
                    if reasoning:
                        lines.append(f"- **Reasoning**: {reasoning}")
                    lines.append("")
                st.markdown("\n".join(lines))
            else:
                st.info("No structured evaluation fields found in this row.")

        with st.expander("Conversation", expanded=False):
            conversation_text = row.get("conversation", "") or ""
            # Render as markdown (the conversation text already uses markdown headings/bold)
            st.markdown(str(conversation_text))

    cols = st.columns(3)
    with cols[1]:
        if st.button("Open details", type="primary", width="stretch"):
            _open_result_dialog()
        if csv_bytes:
            st.download_button(
                label="Download results",
                type="secondary",
                data=csv_bytes,
                file_name="evaluation_results.csv",
                mime="text/csv",
                width="stretch",
            )

    st.markdown("---")
    st.subheader("Summarize results")

    # Ensure base config is loaded so we can access summarizer defaults
    _initialize_config_and_categories_from_base()
    base_config = load_base_config()
    summarizer_system_prompt = str(base_config.get("summarizer_system_prompt", "") or "")

    categories: List[Dict[str, str]] = st.session_state.get("categories") or []
    category_keys = [c.get("key", "") for c in categories if c.get("key")]
    if not category_keys:
        st.info("No categories available to summarize.")
        return

    # Default selected category
    if st.session_state.get("summarizer_selected_category_key") not in category_keys:
        st.session_state["summarizer_selected_category_key"] = category_keys[0]
        _clear_summarizer_outputs()

    def _format_cat(key: str) -> str:
        for c in categories:
            if c.get("key") == key:
                label = c.get("label") or key
                return f"{label}"
        return key

    selected_category_key = st.selectbox(
        "Category",
        options=category_keys,
        key="summarizer_selected_category_key",
        format_func=_format_cat,
        on_change=_clear_summarizer_outputs,
    )

    # Task input (defaulted from config)
    st.session_state["summarizer_task"] = st.text_area(
        "Summarizer task",
        value=st.session_state.get("summarizer_task", ""),
        height=120,
        on_change=_clear_summarizer_outputs,
    )

    # Compute stats for the selected category
    eval_key = f"evaluation_{selected_category_key}"
    reason_key = f"evaluation_{selected_category_key}_reasoning"

    rows_true: List[Dict[str, Any]] = []
    rows_false: List[Dict[str, Any]] = []
    for row in results:
        b = _parse_bool(row.get(eval_key))
        if b is True:
            rows_true.append(row)
        elif b is False:
            rows_false.append(row)

    total_analyzed = len(rows_true) + len(rows_false)
    success_rate_pct = (len(rows_true) / float(total_analyzed) * 100.0) if total_analyzed else 0.0

    avg_user_msgs_true = _mean_int([r.get("userMessageCount") for r in rows_true])
    avg_user_msgs_false = _mean_int([r.get("userMessageCount") for r in rows_false])
    avg_user_chars_true = _mean_int([r.get("avgUserMessageCharacters") for r in rows_true])
    avg_user_chars_false = _mean_int([r.get("avgUserMessageCharacters") for r in rows_false])

    stats_block = {
        "total_conversations_analyzed": total_analyzed,
        "success_rate_percent": round(success_rate_pct, 1),
        "avg_user_messages_true": avg_user_msgs_true,
        "avg_user_messages_false": avg_user_msgs_false,
        "avg_user_chars_true": avg_user_chars_true,
        "avg_user_chars_false": avg_user_chars_false,
    }

    # Show stats immediately (even before calling the model)
    st.markdown(
        f"- **Total conversations analyzed**: `{stats_block['total_conversations_analyzed']}`\n"
        f"- **Success rate**: `{stats_block['success_rate_percent']}%`"
    )

    st.dataframe(
        [
            {
                "Metric": "Avg user messages",
                "Successful Conversations": stats_block["avg_user_messages_true"],
                "Unsuccessful Conversations": stats_block["avg_user_messages_false"],
            },
            {
                "Metric": "Avg user characters",
                "Successful Conversations": stats_block["avg_user_chars_true"],
                "Unsuccessful Conversations": stats_block["avg_user_chars_false"],
            },
        ],
        width="stretch",
        hide_index=True,
    )

    # Summarize button
    max_examples = int(os.getenv("SUMMARIZER_MAX_EXAMPLES", "25"))
    cols2 = st.columns(3)
    with cols2[1]:
        run_summary = st.button("Summarize results", type="primary", width="stretch", disabled=total_analyzed == 0)

    if run_summary:
        if not summarizer_system_prompt:
            st.error("Missing `summarizer_system_prompt` in config.")
            return
        task = (st.session_state.get("summarizer_task") or "").strip()
        if not task:
            st.error("Please provide a summarizer task.")
            return

        # Find description from the editable categories
        category_description = ""
        for c in categories:
            if c.get("key") == selected_category_key:
                category_description = c.get("description", "") or ""
                break

        examples = _select_summary_examples(
            rows_true=rows_true,
            rows_false=rows_false,
            max_examples=max_examples,
        )

        # Build example rows with only the requested fields
        example_rows: List[Dict[str, Any]] = []
        for r in examples:
            example_rows.append(
                {
                    "conversationId": r.get("conversationId", ""),
                    "createdAt": r.get("createdAt", ""),
                    "userMessageCount": r.get("userMessageCount", 0),
                    "providerMessageCount": r.get("providerMessageCount", 0),
                    "avgUserMessageCharacters": r.get("avgUserMessageCharacters", 0),
                    "avgProviderMessageCharacters": r.get("avgProviderMessageCharacters", 0),
                    eval_key: r.get(eval_key),
                    reason_key: r.get(reason_key, ""),
                }
            )

        with st.spinner("Summarizing results..."):
            summarizer = Summarizer(
                SummarizerConfig(
                    system_prompt=summarizer_system_prompt,
                    model_name=(st.session_state.get("config") or {}).get("model_name", ""),
                    config_file=st.session_state.get("config_file_name") or "",
                )
            )
            summary_result = summarizer.summarize_category(
                task=task,
                category_key=selected_category_key,
                category_description=category_description,
                true_count=len(rows_true),
                false_count=len(rows_false),
                success_rate=(len(rows_true) / float(total_analyzed)) if total_analyzed else 0.0,
                examples=example_rows,
                max_tokens=1200,
            )

        st.session_state["summarizer_stats"] = stats_block
        st.session_state["summarizer_output"] = summary_result.text
        st.rerun()

    if st.session_state.get("summarizer_output"):
        st.markdown("### Summary")
        st.markdown(str(st.session_state["summarizer_output"]))


def main() -> None:
    """Streamlit app entry point."""
    
    ensure_session_state_keys()
    render_access_gate()

    st.title("Conversation Evaluation Lab")
    st.write("Run structured evaluations over conversation logs using Claude.")

    st.subheader("Evaluation categories")
    with st.expander("Edit the categories", expanded=False):
        render_categories_editor()

    st.subheader("Evaluator context")
    with st.expander("Add context", expanded=False):
        evaluator_context = st.text_area(
            "Useful background information for the evaluator to improve the assessments",
            value=st.session_state.get("evaluator_context", ""),
            help="Provide any additional context or instructions that should be considered during evaluation.",
            height=350,
        )
        st.session_state["evaluator_context"] = evaluator_context

    st.subheader("Conversation data")
    render_conversation_uploader_and_selector()

    st.subheader("Process conversations")
    ready, error_message = validate_ready_to_run()
    if not ready and error_message:
        st.warning(error_message)

    is_running = st.session_state.get("is_running", False)

    cols = st.columns(3)
    with cols[1]:
        run_button = st.button(
            "Run",
            width="stretch",
            type="primary",
            disabled=not ready or is_running,
        )
    
    logger.debug(f"Run button state: clicked={run_button}, disabled={not ready or is_running}")

    # Initialize thread-safe communication objects
    if st.session_state.get("cancel_event") is None:
        logger.info("Creating new cancel_event")
        st.session_state["cancel_event"] = threading.Event()
    if st.session_state.get("progress_dict") is None:
        logger.info("Creating new progress_dict")
        st.session_state["progress_dict"] = {}

    cancel_event: threading.Event = st.session_state["cancel_event"]
    progress_dict: Dict[str, Any] = st.session_state["progress_dict"]

    # Sync progress_dict to session_state (for display)
    if progress_dict:
        logger.debug(f"Syncing progress_dict to session_state. Keys: {list(progress_dict.keys())}")
        if "progress" in progress_dict:
            st.session_state["progress"] = progress_dict["progress"]
        if "status_text" in progress_dict:
            st.session_state["status_text"] = progress_dict["status_text"]
        if "detail_text" in progress_dict:
            st.session_state["detail_text"] = progress_dict["detail_text"]
        if "results" in progress_dict:
            st.session_state["results"] = progress_dict["results"]
        if "csv_bytes" in progress_dict:
            st.session_state["csv_bytes"] = progress_dict["csv_bytes"]
        if "is_running" in progress_dict:
            st.session_state["is_running"] = progress_dict["is_running"]

    # Start a background worker thread when Run is clicked
    if run_button and ready and not is_running:
        logger.info("Run button clicked, starting worker thread...")
        logger.info(f"ready={ready}, is_running={is_running}")
        
        cancel_event.clear()
        progress_dict.clear()
        progress_dict["is_running"] = True
        st.session_state["is_running"] = True
        logger.info("Cleared cancel_event and initialized progress_dict")

        # Read all data from session_state before starting thread
        config = st.session_state["config"]
        categories = st.session_state["categories"]
        conversations = st.session_state["conversations"]
        num_to_evaluate = st.session_state["num_to_evaluate"]
        sample_randomly = st.session_state["sample_randomly"]
        evaluator_context = st.session_state.get("evaluator_context", "")
        config_file_name = st.session_state.get("config_file_name", "default_config.json")
        
        logger.info(f"Read from session_state: config={bool(config)}, categories={len(categories)}, "
                   f"conversations={len(conversations)}, num_to_evaluate={num_to_evaluate}, "
                   f"sample_randomly={sample_randomly}, evaluator_context length={len(evaluator_context)}, "
                   f"config_file_name={config_file_name}")

        worker = threading.Thread(
            target=_run_evaluations_worker,
            args=(
                config,
                categories,
                conversations,
                num_to_evaluate,
                sample_randomly,
                evaluator_context,
                config_file_name,
                cancel_event,
                progress_dict,
            ),
            daemon=True,
        )
        logger.info("Starting worker thread...")
        worker.start()
        logger.info(f"Worker thread started. Thread ID: {worker.ident}, Alive: {worker.is_alive()}")

        # Immediately rerun to show running state
        logger.info("Triggering rerun...")
        st.rerun()

    # While a run is in progress, show progress and a Cancel button
    if is_running:
        logger.debug(f"Run in progress. Progress dict keys: {list(progress_dict.keys())}")
        progress = float(st.session_state.get("progress", 0.0))
        progress_pct = int(max(0.0, min(1.0, progress)) * 100)
        logger.debug(f"Displaying progress: {progress} ({progress_pct}%)")
        st.progress(progress_pct)

        status_text = st.session_state.get("status_text")
        if status_text:
            st.write(status_text)
            logger.debug(f"Status text: {status_text}")

        detail_text = st.session_state.get("detail_text")
        if detail_text:
            st.caption(detail_text)
            logger.debug(f"Detail text: {detail_text}")

        with cols[2]:
            if st.button("Cancel", type="secondary"):
                logger.info("Cancel button clicked")
                cancel_event.set()
                st.write("Cancellation requested. Finishing current in-flight calls...")
                st.rerun()

        # Auto-refresh UI while background worker is running so we pick up
        # new values written into progress_dict by the thread.
        time.sleep(0.5)
        st.rerun()

    st.markdown("---")
    if st.session_state.get("results"):
        st.subheader("Results")
        render_results_section()


if __name__ == "__main__":
    main()

