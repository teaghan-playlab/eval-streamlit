"""Streamlit app for running conversation evaluations."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
import streamlit as st

from eval_app import (
    Evaluator,
    EvaluatorConfig,
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
st.set_page_config(page_title="Conversation Evaluation Lab", layout="wide")

# User-facing limits (kept in one place for easy adjustment)
MAX_CATEGORIES = int(os.getenv("MAX_CATEGORIES", 10))
MAX_CONVERSATIONS = int(os.getenv("MAX_CONVERSATIONS", 40))


def _project_root() -> Path:
    return Path(__file__).parent


def load_base_config() -> Dict[str, Any]:
    """Load the default config JSON once and cache it."""
    if "base_config" in st.session_state and st.session_state["base_config"] is not None:
        return st.session_state["base_config"]

    config_path = _project_root() / "configs" / os.getenv("DEFAULT_CONFIG_FILE", "default_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    st.session_state["base_config"] = base_config
    return base_config


def ensure_session_state_keys() -> None:
    """Initialize keys the app relies on."""
    defaults = {
        "access_granted": False,
        "base_config": None,
        "config": None,  # {"system_prompt": str, "model_name": str}
        "categories": [],  # [{"label": str, "key": str, "description": str}]
        "conversations": [],
        "num_to_evaluate": None,
        "sample_randomly": False,
        "results": [],
        "csv_bytes": None,
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
    # model name, and categories) as JSON for reuse in future runs.
    cfg = st.session_state.get("config") or {}
    config_payload = {
        "system_prompt": cfg.get("system_prompt", ""),
        "model_name": cfg.get("model_name", ""),
        "categories": categories_to_fields(updated_categories),
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
    for conv in conversations[:preview_count]:
        preview_rows.append(
            {
                "conversationId": conv.get("conversationId", ""),
                "createdAt": conv.get("createdAt", ""),
                "userMessageCount": conv.get("userMessageCount", 0),
                "messageCount": conv.get("messageCount", 0),
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
        "Randomly sample conversations (instead of taking the first N)",
        value=False,
    )


def validate_ready_to_run() -> tuple[bool, str | None]:
    """Check whether all prerequisites to run evaluation are satisfied.
    
    Returns:
        Tuple of (is_valid, error_message). If is_valid is True, error_message is None.
    """
    if not st.session_state.get("access_granted"):
        return False, "Access code is required. Please enter the access code."

    cfg = st.session_state.get("config") or {}
    if not cfg.get("model_name"):
        return False, "Model name is required. Please configure the model name."
    if not cfg.get("system_prompt"):
        return False, "System prompt is required. Please configure the system prompt."

    categories = st.session_state.get("categories") or []
    if len(categories) < 1:
        return False, f"At least 1 category is required. Please add at least one evaluation category."
    if len(categories) > MAX_CATEGORIES:
        return False, f"Too many categories. Maximum {MAX_CATEGORIES} categories are supported. Please remove some categories."

    conversations = st.session_state.get("conversations") or []
    if not conversations:
        return False, "No conversations loaded. Please upload a conversations JSON file."

    num = st.session_state.get("num_to_evaluate")
    if not isinstance(num, int) or num < 1:
        return False, "Invalid number of conversations to evaluate. Please select a valid number."
    if num > min(MAX_CONVERSATIONS, len(conversations)):
        return False, f"Number to evaluate ({num}) exceeds the maximum allowed ({min(MAX_CONVERSATIONS, len(conversations))}). Please select a smaller number."

    if not os.getenv("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY is not set in the environment. Please set the API key."

    return True, None


def run_evaluations_and_collect_results() -> None:
    """Run evaluations on the selected conversations and store results."""
    cfg = st.session_state["config"]
    categories = categories_to_fields(st.session_state["categories"])

    eval_config = EvaluatorConfig(
        system_prompt=cfg["system_prompt"],
        model_name=cfg["model_name"],
        categories=categories,
        config_file="streamlit_inline_config",
    )

    evaluator = Evaluator(eval_config)

    conversations: List[Dict[str, Any]] = st.session_state["conversations"]
    num = st.session_state["num_to_evaluate"]
    sample_randomly = st.session_state["sample_randomly"]

    max_n = min(MAX_CONVERSATIONS, len(conversations))
    num = max(1, min(num, max_n))

    if sample_randomly:
        random.seed()
        selected = random.sample(conversations, num)
    else:
        selected = conversations[:num]

    with st.spinner("Running evaluations..."):
        results = evaluate_conversations(
            evaluator,
            selected,
            conversation_field="conversation",
        )

    st.session_state["results"] = results

    # Build CSV
    headers = order_csv_headers(get_all_headers(results))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp_path = Path(tmp.name)
    write_results_to_csv(results, tmp_path, fieldnames=headers)

    with open(tmp_path, "rb") as f:
        st.session_state["csv_bytes"] = f.read()


def render_results_section() -> None:
    """Render results preview and CSV download controls."""
    results: List[Dict[str, Any]] = st.session_state.get("results") or []
    csv_bytes = st.session_state.get("csv_bytes")

    if not results:
        st.info("No evaluation results yet. Run evaluations to see results here.")
        return

    # Show a small table of key columns (omit full conversation for readability)
    display_rows: List[Dict[str, Any]] = []
    for row in results:
        display_rows.append(
            {
                "conversationId": row.get("conversationId", ""),
                "evaluatorModel": row.get("evaluatorModel", ""),
                "evaluationSuccessful": not row.get("evaluation_decode_failed", False),
            }
        )

    st.dataframe(display_rows, width="stretch")

    if csv_bytes:
        st.download_button(
            label="Download entire evaluation results as CSV",
            type="primary",
            data=csv_bytes,
            file_name="evaluation_results.csv",
            mime="text/csv",
        )


def main() -> None:
    """Streamlit app entry point."""
    ensure_session_state_keys()
    render_access_gate()

    st.title("Conversation Evaluation Lab")
    st.write("Run structured evaluations over conversation logs using Anthropic.")

    st.subheader("Define evaluation categories")
    with st.expander("Edit the categories", expanded=False):
        render_categories_editor()

    st.subheader("Upload conversations")
    render_conversation_uploader_and_selector()

    st.subheader("Run evaluations")
    ready, error_message = validate_ready_to_run()
    if not ready and error_message:
        st.warning(error_message)
    with st.columns(3)[1]:
        run_button = st.button("Run evaluations", width="stretch", type="primary", disabled=not ready)
    if run_button and ready:
        run_evaluations_and_collect_results()

    if st.session_state.get("results"):
        st.subheader("Download results")
        render_results_section()


if __name__ == "__main__":
    main()

