"""Typed JSON output with an explicit, host-authorized completion adapter.

No provider is selected implicitly. Construct StructuredOutputEngine(completion_fn)
where the async adapter accepts keyword-only model and messages and returns
text, actual model, cost_usd and latency_ms. The existing orchestrator's
think(task, context) contract cannot supply this guarantee.

Function calling validates a proposed call; it never executes a function.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from referencing import Registry

T = TypeVar("T", bound=BaseModel)
CompletionFn = Callable[..., Awaitable[Mapping[str, Any]]]


class StructuredBackendNotConfigured(RuntimeError):
    """The host has not supplied a model-aware completion adapter."""


class StructuredBackendProtocolError(RuntimeError):
    """The adapter omitted the actual response or measurement metadata."""


@dataclass
class StructuredResult:
    data: Any
    raw_text: str
    parsed: bool
    validation_errors: List[str]
    retry_count: int
    model: str
    cost_usd: float
    latency_ms: float


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _deny_retrieval(uri):
    raise ValueError("External JSON Schema references are disabled")


def _local_schema_only(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef", "$recursiveRef", "$id"}:
                if not isinstance(child, str) or (child and not child.startswith("#")):
                    raise ValueError("Only local JSON Schema references are supported")
            _local_schema_only(child)
    elif isinstance(value, list):
        for child in value:
            _local_schema_only(child)


def _function_parameters(parameters):
    """Accept standard object schemas and the legacy property-map format."""
    if not isinstance(parameters, dict):
        raise ValueError("Function parameters must be an object")
    if parameters.get("type") == "object" or "properties" in parameters or "$schema" in parameters:
        # Copy the caller's schema; never rewrite its nested constraints.
        schema = json.loads(json.dumps(parameters))
        if schema.get("type") != "object":
            raise ValueError("Function parameter schemas must declare type=object")
        if "$schema" in schema and schema["$schema"] != "https://json-schema.org/draft/2020-12/schema":
            raise ValueError("Function parameter schemas must use JSON Schema 2020-12")
        schema.setdefault("additionalProperties", False)
    else:
        properties, required = {}, []
        for name, definition in parameters.items():
            if not isinstance(definition, dict):
                raise ValueError(f"Invalid parameter definition: {name}")
            definition = dict(definition)
            is_required = definition.pop("required", False)
            if not isinstance(is_required, bool):
                raise ValueError("Legacy parameter required must be a boolean")
            if is_required:
                required.append(name)
            properties[name] = definition
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    _local_schema_only(schema)
    Draft202012Validator.check_schema(schema)
    return schema


class _FunctionCallSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    function: str
    arguments: Dict[str, Any]


class StructuredOutputEngine:
    MAX_RETRIES = 3

    def __init__(self, completion_fn: Optional[CompletionFn] = None):
        if completion_fn is not None and not callable(completion_fn):
            raise TypeError("completion_fn must be callable")
        self._completion_fn = completion_fn

    def _generate_schema_prompt(self, schema: Type[BaseModel]) -> str:
        return (
            "Respond with JSON only, without prose or markdown. Match this JSON Schema:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False, sort_keys=True)
        )

    def _extract_json(self, text: str) -> str:
        # A single fenced JSON value is accepted, but surrounding prose, duplicate
        # keys and malformed JSON are not silently repaired or reinterpreted.
        text = text.strip()
        fence = chr(96) * 3
        fenced = re.fullmatch(fence + r"(?:json)?\s*\n?(.*?)\n?" + fence, text, re.DOTALL)
        return fenced.group(1).strip() if fenced else text

    async def infer(
        self,
        prompt: str,
        schema: Type[T],
        model: str = "deepseek-v4-pro",
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> StructuredResult:
        """Validate responses, with at most max_retries total provider attempts.

        The legacy parameter name is retained; retry_count reports retries
        actually made after the first attempt. Provider/adapter errors propagate
        immediately and are not disguised as invalid model JSON.
        """
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 1:
            raise ValueError("max_retries must be a positive integer")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty")
        if self._completion_fn is None:
            raise StructuredBackendNotConfigured(
                "Supply completion_fn(*, model, messages); think(task, context) "
                "cannot honor explicit model and system_prompt."
            )

        schema_instruction = self._generate_schema_prompt(schema)
        system = f"{system_prompt}\n\n{schema_instruction}" if system_prompt else schema_instruction
        user_prompt = prompt
        last_error = ""
        raw_text = ""
        actual_model = ""
        total_cost = 0.0
        total_latency = 0.0

        for attempt in range(max_retries):
            result = await self._completion_fn(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if (
                not isinstance(result, Mapping)
                or not isinstance(result.get("text"), str)
                or not isinstance(result.get("model"), str)
                or not result["model"].strip()
            ):
                raise StructuredBackendProtocolError("Adapter must return text and actual model")
            for key in ("cost_usd", "latency_ms"):
                value = result.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise StructuredBackendProtocolError(f"Adapter must return nonnegative finite {key}")
            raw_text = result["text"]
            actual_model = result["model"]
            total_cost += result["cost_usd"]
            total_latency += result["latency_ms"]
            try:
                data = json.loads(
                    self._extract_json(raw_text),
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
                validated = schema.model_validate(data, strict=True)
                return StructuredResult(
                    data=validated, raw_text=raw_text, parsed=True,
                    validation_errors=[], retry_count=attempt, model=actual_model,
                    cost_usd=total_cost, latency_ms=total_latency,
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                user_prompt = (
                    prompt + "\n\nThe previous response failed validation:\n"
                    + last_error + "\nReturn only a corrected JSON value."
                )

        return StructuredResult(
            data=None, raw_text=raw_text, parsed=False,
            validation_errors=[last_error], retry_count=max_retries - 1,
            model=actual_model, cost_usd=total_cost, latency_ms=total_latency,
        )

    async def function_call(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        model: str = "deepseek-v4-pro",
    ) -> Dict[str, Any]:
        """Return a validated proposed call, never execute it.

        Supports JSON Schema 2020-12 object parameters and the original
        {parameter: {type: ..., required: bool}} format. Unknown function names,
        missing/extra arguments, wrong types, enums and nested constraints fail
        validation. External schema references are never retrieved.
        """
        if not functions:
            raise ValueError("At least one function is required")
        validators, descriptions = {}, []
        for function in functions:
            name = function.get("name")
            if not isinstance(name, str) or not name.strip() or name in validators:
                raise ValueError("Function names must be non-empty and unique")
            parameters = _function_parameters(function.get("parameters", {}))
            validators[name] = Draft202012Validator(
                parameters, format_checker=FormatChecker(),
                registry=Registry(retrieve=_deny_retrieval),
            )
            descriptions.append({
                "name": name,
                "description": function.get("description", ""),
                "parameters": parameters,
            })

        class FunctionCall(_FunctionCallSchema):
            @model_validator(mode="after")
            def validate_selected_function(self):
                if self.function not in validators:
                    raise ValueError(f"Unknown function: {self.function}")
                try:
                    validators[self.function].validate(self.arguments)
                except JSONSchemaValidationError as exc:
                    raise ValueError(f"Invalid function arguments: {exc.message}") from exc
                return self

        result = await self.infer(
            prompt=prompt,
            schema=FunctionCall,
            model=model,
            system_prompt=(
                "Propose one call using the exact function name and valid arguments. "
                "Available functions:\n" + json.dumps(descriptions, ensure_ascii=False, sort_keys=True)
            ),
        )
        if result.parsed:
            return result.data.model_dump()
        return {
            "error": "Failed to validate function call",
            "raw": result.raw_text,
            "validation_errors": result.validation_errors,
        }


# Deliberately unconfigured: the host must supply an authorized model dispatcher.
structured_engine = StructuredOutputEngine()
