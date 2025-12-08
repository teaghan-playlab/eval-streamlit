import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Union

from anthropic import Anthropic, transform_schema
from pydantic import BaseModel, Field, TypeAdapter, create_model


@dataclass
class EvaluationResult:
    """
    Result of an evaluation run.
    - model: model name actually used by the provider
    - config_file: path to the config file used
    - decode_failed: whether JSON parsing failed
    - full_response: the full parsed response payload
    """

    model: str
    config_file: str
    decode_failed: bool = False
    full_response: Optional[Dict[str, Any]] = None


@dataclass
class EvaluatorConfig:
    """Configuration for the Evaluator."""
    system_prompt: str  # System prompt text (can be provided directly or loaded from file)
    model_name: str
    categories: Dict[str, Dict[str, Any]]  # Category definitions: {"category_name": {"description": "..."}}
    system_prompt_path: Optional[str] = None  # Optional: path to system prompt file (if system_prompt not provided)
    base_url: Optional[str] = None  # Not used with direct Anthropic API, kept for compatibility
    config_file: Optional[str] = None  # Path to config file for tracking


class Evaluator:
    """
    Evaluator that requests strict JSON schema output from Anthropic Claude API.
    Uses Anthropic's structured outputs feature for reliable JSON responses.
    """

    # Beta header required for structured outputs feature
    STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"

    def __init__(
        self,
        config: EvaluatorConfig,
        client: Optional[Anthropic] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the Evaluator.

        Args:
            config: Configuration object with system prompt path, model name, and category definitions
            client: Optional pre-configured Anthropic client. If not provided, one will be created.
            api_key: Optional API key. If not provided, will use ANTHROPIC_API_KEY env var.
        """
        # Initialize Anthropic client
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

        # Load system prompt (from direct field or file)
        if config.system_prompt:
            self.system_prompt = config.system_prompt
        elif config.system_prompt_path:
            system_prompt_path = Path(config.system_prompt_path)
            if not system_prompt_path.exists():
                raise FileNotFoundError(f"System prompt file not found: {config.system_prompt_path}")
            with open(system_prompt_path, "r", encoding="utf-8") as fh:
                self.system_prompt = fh.read()
        else:
            raise ValueError("Either system_prompt or system_prompt_path must be provided in config")

        self.model_name = config.model_name
        self.config_file = config.config_file or ""
        self.temperature = 0.0  # Use deterministic temperature for evaluations
        self.last_payload: Optional[BaseModel] = None

        # Dynamically create the response model from categories
        self.response_model = self._create_response_model(config.categories)
        
        # Pre-compute the output format for Anthropic API
        self.output_format = self._get_output_format()

        logging.info(
            f"Initialized Evaluator with model {self.model_name} "
            f"and {len(config.categories)} evaluation categories"
        )

    def _create_response_model(self, categories: Dict[str, Dict[str, Any]]) -> Type[BaseModel]:
        """
        Dynamically create a Pydantic model from category definitions.
        Automatically generates reasoning fields for each category (all categories are boolean).

        Args:
            categories: Dictionary mapping category names to their definitions.
                       Each definition should have a 'description' key.

        Returns:
            A Pydantic BaseModel class with the specified fields.
        """
        field_definitions = {}

        # Process categories in order to maintain schema field order
        # For each category, create a reasoning field first, then the boolean field
        for category_name, category_config in categories.items():
            description = category_config.get("description", f"Category {category_name}")
            
            # Create reasoning field first
            reasoning_field_name = f"{category_name}_reasoning"
            reasoning_description = (
                f"A concise (one sentence) reasoning statement for your evaluation of the {category_name} category based on the criteria provided."
            )
            
            # Add reasoning field (required string to avoid anyOf in schema)
            field_definitions[reasoning_field_name] = (
                str,
                Field(..., description=reasoning_description)
            )
            
            # Then add the boolean category field (required)
            field_definitions[category_name] = (
                bool,
                Field(..., description=description)
            )

        # Create the model dynamically
        return create_model("EvaluationResponse", **field_definitions)

    def _get_output_format(self) -> Dict[str, Any]:
        """
        Get the output format dictionary for Anthropic's structured outputs.

        Returns:
            Output format dictionary ready for Anthropic API with type and schema.
        """
        # Get JSON schema from Pydantic model
        type_adapter = TypeAdapter(self.response_model)
        json_schema = type_adapter.json_schema()
        
        # Transform schema for Anthropic API compatibility
        transformed_schema = transform_schema(json_schema)
        
        # Return in the format expected by Anthropic API
        return {
            "type": "json_schema",
            "schema": transformed_schema,
        }

    def evaluate(self, text: str, use_prompt_caching: bool = False) -> EvaluationResult:
        """
        Run evaluation on the given text.

        Args:
            text: The raw text to evaluate.
            source: Optional source indicator:
                - "user": prefix content with "User message:\\n\\n"
                - "provider": prefix content with "AI chatbot message:\\n\\n"
                - "settings": prefix content with "Chatbot Settings:\\n\\n"
                - None/other: no prefix applied.

        Returns:
            EvaluationResult with the evaluation outcome.
        """

        fallback = EvaluationResult(
            model=self.model_name,
            config_file=self.config_file,
            decode_failed=True,
            full_response=None,
        )

        try:
            return self._evaluate_with_schema(text, use_prompt_caching=use_prompt_caching)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Structured evaluation failed: %s", exc)

            # Handle content policy errors
            try:
                error_message = str(exc)
                is_content_policy_error = (
                    "content_policy_violation" in error_message.lower()
                    or "Response output was blocked" in error_message
                )
                if is_content_policy_error:
                    logging.warning(
                        f"Content policy violation for model {self.model_name}: {error_message}"
                    )
            except Exception:
                # Never let logging failures break evaluation
                logging.exception("Failed to log content policy violation")

            return fallback

    def _evaluate_with_schema(self, text: str, use_prompt_caching: bool = False) -> EvaluationResult:
        """
        Perform evaluation using Anthropic's structured outputs.

        Args:
            text: The text to evaluate (may include source prefix).

        Returns:
            EvaluationResult with parsed response.
        """
        try:
            # Debug logging: Schema
            logging.debug("=" * 60)
            logging.debug("EVALUATION DEBUG INFO")
            logging.debug("=" * 60)
            logging.debug("\nSchema (output_format):")
            logging.debug(json.dumps(self.output_format, indent=2))

            # Debug logging: System prompt
            logging.debug("\nSystem Prompt:")
            logging.debug("-" * 60)
            logging.debug(self.system_prompt)
            logging.debug("-" * 60)

            # Debug logging: Input message (truncated)
            input_preview = text[:1000] + "..." if len(text) > 1000 else text
            logging.debug(f"\nInput Message (truncated to 1000 chars, total: {len(text)} chars):")
            logging.debug("-" * 60)
            logging.debug(input_preview)
            logging.debug("-" * 60)

            # Make API call with structured outputs.
            # Optionally use Anthropic prompt caching for the (potentially large) shared system prompt
            # by converting the plain string system prompt into a content block with cache_control.
            if use_prompt_caching:
                system = [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                system = self.system_prompt
            response = self.client.beta.messages.create(
                model=self.model_name,
                max_tokens=4096,  # Reasonable default, can be made configurable
                temperature=self.temperature,
                betas=[self.STRUCTURED_OUTPUTS_BETA],
                output_format=self.output_format,
                system=system,
                messages=[
                    {"role": "user", "content": text}
                ],
            )

            # Extract the structured response
            # Anthropic returns structured output in response.content[0].text
            if not response.content:
                raise ValueError("Empty response from Anthropic API")
            
            response_text = response.content[0].text

            # Debug logging: Response
            logging.debug("\nAPI Response:")
            logging.debug("-" * 60)
            logging.debug(response_text)
            logging.debug("-" * 60)
            logging.debug("=" * 60)
            
            # Parse JSON response
            try:
                parsed_json = json.loads(response_text)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response: {response_text[:200]}")
                raise ValueError(f"Invalid JSON in response: {e}") from e

            # Clean up any nested structures (defensive parsing)
            cleaned_json = self._clean_response(parsed_json)

            # Validate with Pydantic model
            payload = self.response_model.model_validate(cleaned_json)
            self.last_payload = payload

            # Get model name from response (may differ if model was aliased)
            model_used = getattr(response, "model", self.model_name)

            # Debug logging: Parsed result
            logging.debug("\nParsed Evaluation Result:")
            logging.debug(json.dumps(payload.model_dump(), indent=2))

            return EvaluationResult(
                model=model_used,
                config_file=self.config_file,
                decode_failed=False,
                full_response=payload.model_dump(),
            )

        except Exception as e:
            logging.error(f"Anthropic API call failed: {e}")
            raise

    def _clean_response(self, obj: Any) -> Any:
        """
        Clean up response JSON by unwrapping common nested structures.
        This is defensive parsing to handle edge cases.

        Args:
            obj: The parsed JSON object to clean.

        Returns:
            Cleaned object.
        """
        # Apply all cleaning functions
        return obj


__all__ = ["Evaluator", "EvaluatorConfig", "EvaluationResult"]
