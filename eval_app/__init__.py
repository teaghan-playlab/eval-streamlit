"""Core evaluation package for the eval-streamlit project.

This package exposes the main building blocks used by both the CLI
(`scripts/run_evaluations.py`) and the Streamlit app (`app.py`).

Public API (intended for use by callers):
- Evaluator, EvaluatorConfig, EvaluationResult  (from .evaluation)
- load_config                                  (from .config_loader)
- evaluate_conversations                       (from .runner)
- load_conversations_from_file, load_all_conversations,
  get_all_headers, order_csv_headers           (from .data_loader)
- write_results_to_csv, append_results_to_csv  (from .results_tracker)
- DigDeeper, DigDeeperConfig, DigDeeperResult  (from .dig_deeper)
"""

from .evaluation import Evaluator, EvaluatorConfig, EvaluationResult
from .config_loader import (
    load_config,
    fields_to_categories,
    categories_to_fields,
    slugify_label,
)
from .runner import evaluate_conversations
from .data_loader import (
    load_conversations_from_file,
    load_all_conversations,
    get_all_headers,
    order_csv_headers,
)
from .results_tracker import write_results_to_csv, append_results_to_csv
from .dig_deeper import DigDeeper, DigDeeperConfig, DigDeeperResult

__all__ = [
    "Evaluator",
    "EvaluatorConfig",
    "EvaluationResult",
    "DigDeeper",
    "DigDeeperConfig",
    "DigDeeperResult",
    "load_config",
    "fields_to_categories",
    "categories_to_fields",
    "slugify_label",
    "evaluate_conversations",
    "load_conversations_from_file",
    "load_all_conversations",
    "get_all_headers",
    "order_csv_headers",
    "write_results_to_csv",
    "append_results_to_csv",
]
